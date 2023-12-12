use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferUsage, RecordingCommandBuffer, CopyBufferToImageInfo, PrimaryAutoCommandBuffer, CopyBufferInfo,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags, Queue, Features,
    },
    format::Format,
    image::{Image, ImageType, ImageUsage, ImageCreateInfo, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
    VulkanLibrary,
};

fn initialize_resources() -> (Arc<Queue>, Arc<Device>) {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )
    .unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
    .enumerate_physical_devices()
    .unwrap()
    .filter(|p| p.supported_extensions().contains(&device_extensions))
    .filter_map(|p| {
        p.queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
            .map(|i| (p, i as u32))
    })
    .min_by_key(|(p, _)| match p.properties().device_type {
        PhysicalDeviceType::DiscreteGpu => 0,
        PhysicalDeviceType::IntegratedGpu => 1,
        PhysicalDeviceType::VirtualGpu => 2,
        PhysicalDeviceType::Cpu => 3,
        PhysicalDeviceType::Other => 4,
        _ => 5,
    })
    .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    for (index, heap) in physical_device.memory_properties().memory_heaps.iter().enumerate() {
        println!("Heap #{:?} has a capacity of {:?} bytes with flags {:?}", index, heap.size, heap.flags);
    }

    for ty in physical_device.memory_properties().memory_types.iter() {
        println!("Memory type belongs to heap #{:?}, with flags: {:?}", ty.heap_index, ty.property_flags);
    }

    let features = Features {
        storage_buffer16_bit_access: true,
        shader_int16: true,
        .. Features::default()
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            enabled_features: features,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    (queue, device)
}

struct DefectMapBufferResources {
    pipeline: Arc<ComputePipeline>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,    
    kernel_buffer: Subbuffer<[u16]>,
    defect_map_buffer: Subbuffer<[u16]>,
}

impl DefectMapBufferResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, 
        memory_allocator: Arc<StandardMemoryAllocator>, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>, image_height: u32, image_width: u32) -> Self {
            let pipeline = {
                mod offset_correction_shader {
                    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
                            #version 450
                            #extension GL_EXT_shader_16bit_storage : require
                            #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

                            #define KERNEL_SIZE 5
        
                            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                            layout(set = 0, binding = 0) buffer DefectData {
                                uint16_t defectMapData[];
                            };

                            layout(set = 0, binding = 1) buffer ImageData {
                                uint16_t imageData[];
                            };

                            layout(set = 0, binding = 2) buffer ResultImage {
                                uint16_t resultData[];
                            };
                            
                            layout(set = 0, binding = 3) buffer KernelData {
                                uint16_t kernelData[KERNEL_SIZE];
                            };
        
                            void main() {
                                uint idx = gl_GlobalInvocationID.x;
                                uint16_t sum = uint16_t(0);

                                for (int i = 0; i < KERNEL_SIZE; ++i) {
                                    int imageIndex = int(idx) - KERNEL_SIZE / 2 + i;

                                    if (imageIndex >= 0 && imageIndex < imageData.length()) {
                                        sum += imageData[imageIndex] * kernelData[i];
                                    }
                                }


                            }
                        ",
                    }
                }
                
            let cs = offset_correction_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
            };

            let defect_map_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![0u16; (image_width * image_height) as usize] /* number of elements, matching the image size */,
            )
            .unwrap();

            let kernel_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![1u16, 2, 3, 2, 1]
            ).unwrap();

            let builder = RecordingCommandBuffer::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let command_buffer = builder.end().unwrap();
    
            let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        
            future.wait(None).unwrap();
    
            DefectMapBufferResources {
                pipeline,
                memory_allocator,
                descriptor_set_allocator,
                defect_map_buffer,
                kernel_buffer
            }
        }

        fn apply_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, image_width: u32, image_height: u32, image_buffer: Subbuffer<[u16]>) {
            let local_size_x = 64;
            
            let dispatch_size_x = (image_width * image_height + local_size_x - 1) / local_size_x;
        
            let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, self.defect_map_buffer.clone()), WriteDescriptorSet::buffer(1, image_buffer.clone())],
                [],
            )
            .unwrap();
        
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .dispatch([dispatch_size_x, 1, 1])
                .unwrap();   
        }
}

struct GainMapBufferResources {
    pipeline: Arc<ComputePipeline>,
    gain_map_buffer: Subbuffer<[f32]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>
}

impl GainMapBufferResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, 
        memory_allocator: Arc<StandardMemoryAllocator>, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>, image_height: u32, image_width: u32) -> Self {
            let pipeline = {
                mod offset_correction_shader {
                    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
                            #version 450
                            #extension GL_EXT_shader_16bit_storage : require
                            #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
        
                            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                            layout(set = 0, binding = 0) buffer GainMapData {
                                float gainMapData[];
                            };
                            layout(set = 0, binding = 1) buffer ImageData {
                                uint16_t imageData[];
                            };
        
                            void main() {
                                uint idx = gl_GlobalInvocationID.x;
                                uint16_t new_val = uint16_t(float(imageData[idx]) * gainMapData[idx]);
                                imageData[idx] = new_val;
                            }
                        ",
                    }
                }
                
            let cs = offset_correction_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
            };

            let gain_map_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![0.5f32; (image_width * image_height) as usize] /* number of elements, matching the image size */,
            )
            .unwrap();

            let builder = RecordingCommandBuffer::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            
            let command_buffer = builder.end().unwrap();
    
            let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        
            future.wait(None).unwrap();
    
            GainMapBufferResources {
                pipeline,
                gain_map_buffer,
                memory_allocator,
                descriptor_set_allocator
            }
        }

        fn apply_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, image_width: u32, image_height: u32, image_buffer: Subbuffer<[u16]>) {
            let local_size_x = 64;
            
            let dispatch_size_x = (image_width * image_height + local_size_x - 1) / local_size_x;
        
            let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
            let set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                layout.clone(),
                [WriteDescriptorSet::buffer(0, self.gain_map_buffer.clone()), WriteDescriptorSet::buffer(1, image_buffer)],
                [],
            )
            .unwrap();
        
            builder
                .bind_pipeline_compute(self.pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .dispatch([dispatch_size_x, 1, 1])
                .unwrap();   
        }
}

struct DarkMapBufferResources {
    pipeline: Arc<ComputePipeline>,
    dark_map_buffer: Subbuffer<[u16]>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DarkMapBufferResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, 
        memory_allocator: Arc<StandardMemoryAllocator>, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>, image_height: u32, image_width: u32) -> Self {
            let pipeline = {
                mod offset_correction_shader {
                    vulkano_shaders::shader! {
                        ty: "compute",
                        src: r"
                            #version 450
                            #extension GL_EXT_shader_16bit_storage : require
                            #extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

                            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                            layout(set = 0, binding = 0) buffer DarkMapData {
                                uint16_t darkMapData[];
                            };
                            layout(set = 0, binding = 1) buffer ImageData {
                                uint16_t imageData[];
                            };
        
                            void main() {
                                uint idx = gl_GlobalInvocationID.x;
                                imageData[idx] = darkMapData[idx] + uint16_t(300);
                            }
                        ",
                    }
                }
                
            let cs = offset_correction_shader::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
            };
    
            let dark_map_buffer = Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                vec![0u16; (image_width * image_height) as usize] /* number of elements, matching the image size */,
            )
            .unwrap();

            let builder = RecordingCommandBuffer::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            
            let command_buffer = builder.end().unwrap();
    
            let future = sync::now(device)
            .then_execute(queue, command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        
            future.wait(None).unwrap();
    
            DarkMapBufferResources {
                pipeline,
                dark_map_buffer,
                memory_allocator,
                descriptor_set_allocator
            }
    }

    fn apply_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, image_width: u32, image_height: u32, image_buffer: Subbuffer<[u16]>) {
        let local_size_x = 64;
            
        let dispatch_size_x = (image_width * image_height + local_size_x - 1) / local_size_x;
    
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.dark_map_buffer.clone()), WriteDescriptorSet::buffer(1, image_buffer)],
            [],
        )
        .unwrap();
    
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([dispatch_size_x, 1, 1])
            .unwrap();   
        }
}

struct DarkMapResources {
    pipeline: Arc<ComputePipeline>,
    dark_map: Arc<Image>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DarkMapResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, 
        memory_allocator: Arc<StandardMemoryAllocator>, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>, image_height: u32, image_width: u32) -> Self {

        let pipeline = {
            mod offset_correction_shader {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
                        #version 450
    
                        layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
    
                        layout(set = 0, binding = 0, r16ui) uniform uimage2D tex;
                        layout(set = 0, binding = 1, r16ui) uniform uimage2D dark_map_tex;
    
                        void main() {
                            ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
                            vec4 val = imageLoad(tex, coords);
                            vec4 dark_map_val = imageLoad(dark_map_tex, coords);
                            val.r  = min(max(val.r - dark_map_val.r + 300, 0), 16383);
                            imageStore(tex, coords, uvec4(val.r, 0, 0, 1));
                        }
                    ",
                }
            }
            
        let cs = offset_correction_shader::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
        };

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u16; (image_height*image_width) as usize],
        )
        .unwrap();


        let dark_map = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16_UINT,
                extent: [image_width as u32, image_height as u32, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST, ..Default::default()
            }
        ).unwrap();

        let mut builder = RecordingCommandBuffer::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(upload_buffer.clone(), dark_map.clone())).unwrap();

        let command_buffer = builder.end().unwrap();

        let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
        future.wait(None).unwrap();

        DarkMapResources {
            pipeline,
            dark_map,
            memory_allocator,
            descriptor_set_allocator
        }
    }

    fn apply_defect_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, device: Arc<Device>, queue: Arc<Queue>, image_width: u32, image_height: u32, image_view: Arc<ImageView>) {
        let local_size_x = 32;
        let local_size_y = 32;
            
        let dispatch_size_x = (image_width + local_size_x - 1) / local_size_x; 
        let dispatch_size_y = (image_height + local_size_y - 1) / local_size_y; 

        let dark_map_view = ImageView::new_default(self.dark_map.clone()).unwrap();
    
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::image_view(0, image_view), WriteDescriptorSet::image_view(1, dark_map_view)],
            [],
        )
        .unwrap();
    
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([dispatch_size_x as u32, dispatch_size_y as u32, 1])
            .unwrap();   
        }
    }

struct DefectMapResources {
    pipeline: Arc<ComputePipeline>,
    defect_map: Arc<Image>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl DefectMapResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, 
        memory_allocator: Arc<StandardMemoryAllocator>, descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>, image_height: u32, image_width: u32) -> Self {
        let pipeline = {
            mod offset_correction_shader {
                vulkano_shaders::shader! {
                    ty: "compute",
                    src: r"
                        #version 450
    
                        layout(set = 0, binding = 0, r16ui) uniform readonly uimage2D defect_map;
                        layout(set = 0, binding = 1, r16ui) uniform uimage2D input_image;
                        layout(set = 0, binding = 2) buffer Kernel {
                            uint kernel[];
                        };
    
                        layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
    
                        ivec2 get_pass_coords(int x, int y, int offset) {
                            if (1 == 1) {
                                return ivec2(x, y + offset); // Vertical pass
                            } else {
                                return ivec2(x + offset, y); // Horizontal pass
                            }
                        }
    
                        void main() {
                            ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
                            vec4 val = imageLoad(input_image, coords);
                            vec4 val_defect = imageLoad(defect_map, coords);
                            uint kernel_val = kernel[0];
                        }
                    ",
                }
            }
            
        let cs = offset_correction_shader::load(device.clone())
        .unwrap()
        .entry_point("main")
        .unwrap();
        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
    
        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )
        .unwrap()
        };

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vec![0u16; (image_height*image_width) as usize],
        )
        .unwrap();


        let defect_map = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16_UINT,
                extent: [image_width as u32, image_height as u32, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST, ..Default::default()
            }
        ).unwrap();

        let mut builder = RecordingCommandBuffer::primary(
            command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(upload_buffer.clone(), defect_map.clone())).unwrap();

        let command_buffer = builder.end().unwrap();

        let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    
        future.wait(None).unwrap();

        DefectMapResources {
            pipeline,
            defect_map,
            memory_allocator,
            descriptor_set_allocator
        }
    }

    fn apply_defect_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, device: Arc<Device>, queue: Arc<Queue>, image_width: u32, image_height: u32, image_view: Arc<ImageView>) {
        let local_size_x = 32;
        let local_size_y = 32;
            
        let dispatch_size_x = (image_width + local_size_x - 1) / local_size_x; 
        let dispatch_size_y = (image_height + local_size_y - 1) / local_size_y; 

        let defect_map_view = ImageView::new_default(self.defect_map.clone()).unwrap();

        let kernel_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [1, 2, 3, 2, 1, 1]
        ).unwrap();    
    
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [WriteDescriptorSet::image_view(0, image_view), WriteDescriptorSet::image_view(1, defect_map_view), WriteDescriptorSet::buffer(2, kernel_buffer)],
            [],
        )
        .unwrap();
    
        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([dispatch_size_x as u32, dispatch_size_y as u32, 1])
            .unwrap();   
        }
    }

fn main() {
    let (queue, device) = initialize_resources();

    let image_width = 4800;
    let image_height = 5800;
    let local_size_x = 32;
    let local_size_y = 32;

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    )); 

    let readback_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        vec![0u16; (image_width*image_height) as usize] /* number of elements, matching the image size */,
    )
    .unwrap();

    //let defect_resources: DefectMapResources = DefectMapResources::new(device.clone(), queue.clone(), command_buffer_allocator.clone(),  memory_allocator.clone(), descriptor_set_allocator.clone(), image_height as u32, image_width as u32);
    let dark_map_resources = DarkMapBufferResources::new(device.clone(), queue.clone(), command_buffer_allocator.clone(), memory_allocator.clone(), descriptor_set_allocator.clone(), image_height as u32, image_width as u32);
    let gain_map_resources = GainMapBufferResources::new(device.clone(), queue.clone(), command_buffer_allocator.clone(), memory_allocator.clone(), descriptor_set_allocator.clone(), image_height as u32, image_width as u32);

    let mut builder = RecordingCommandBuffer::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R16_UINT,
            extent: [image_width as u32, image_height as u32, 1],
            usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE, ..Default::default()
        }
    ).unwrap();

    let image_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![10u16; (image_width*image_height) as usize] /* number of elements, matching the image size */,
    )
    .unwrap();

    println!("hello");

    //builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(image_upload_buffer.clone(), image.clone())).unwrap();
    //let image_view: Arc<ImageView> = ImageView::new_default(image.clone()).unwrap();

    {
        let time = std::time::Instant::now();
        //defect_resources.apply_defect_pipeline(&mut builder, device.clone(), queue.clone(), image_width, image_height, image_view.clone());
        dark_map_resources.apply_pipeline(&mut builder, image_width as u32, image_height as u32, image_buffer.clone());
        gain_map_resources.apply_pipeline(&mut builder, image_width, image_height, image_buffer.clone());
        println!("{:?}", time.elapsed());
    }

    /*
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image.clone(), readback_buffer.clone()))
        .unwrap();
    */

    builder.copy_buffer(CopyBufferInfo::buffers(image_buffer.clone(), readback_buffer.clone())).unwrap();

    let command_buffer = builder.end().unwrap();
    let time = std::time::Instant::now();

    let future = sync::now(device)
    .then_execute(queue, command_buffer)
    .unwrap()
    .then_signal_fence_and_flush()
    .unwrap();

    future.wait(None).unwrap();

    let data_buffer_content = readback_buffer.read().unwrap().to_vec();
    println!("{}", data_buffer_content.last().unwrap());
    println!("Total time {:?}", time.elapsed());


    println!("Success");
}
