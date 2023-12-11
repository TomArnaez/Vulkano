use std::{sync::Arc, default};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, CommandBufferAllocator}, CommandBufferUsage, RecordingCommandBuffer, CopyImageToBufferInfo, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags, Queue,
    },
    format::Format,
    image::{Image, ImageType, ImageUsage, ImageCreateInfo, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::{allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator}, DeviceMemory, MemoryAllocateInfo},
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

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
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

fn apply_pipeline(builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, device: Arc<Device>, queue: Arc<Queue>, image_width: u32, image_height: u32, image_view: Arc<ImageView>) {
    let local_size_x = 32;
    let local_size_y = 32;
        
    let dispatch_size_x = (image_width + local_size_x - 1) / local_size_x; 
    let dispatch_size_y = (image_height + local_size_y - 1) / local_size_y; 

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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

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
        // Iterator that produces the data.
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
            memory_type_filter: MemoryTypeFilter::PREFER_HOST , ..Default::default()
        }
    ).unwrap();

    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(upload_buffer.clone(), dark_map.clone())).unwrap();

    let dark_map_view = ImageView::new_default(dark_map.clone()).unwrap();
    
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = DescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view), WriteDescriptorSet::image_view(1, dark_map_view)],
        [],
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap()
        .dispatch([dispatch_size_x as u32, dispatch_size_y as u32, 1])
        .unwrap();
}

struct DefectMapResources {
    pipeline: Arc<ComputePipeline>,
    defect_map: Arc<Image>,
}

impl DefectMapResources {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, command_buffer_allocator: Arc<StandardCommandBufferAllocator>, image_height: u32, image_width: u32) -> Self {
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

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

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

        println!("defect map memory {:?}", upload_buffer.buffer().memory_requirements());

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
            defect_map
        }
    }

    
fn apply_defect_pipeline(&self, builder: &mut RecordingCommandBuffer<PrimaryAutoCommandBuffer>, device: Arc<Device>, queue: Arc<Queue>, image_width: u32, image_height: u32, image_view: Arc<ImageView>) {
    let local_size_x = 32;
    let local_size_y = 32;
        
    let dispatch_size_x = (image_width + local_size_x - 1) / local_size_x; 
    let dispatch_size_y = (image_height + local_size_y - 1) / local_size_y; 

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let kernel_buffer = Buffer::from_iter(
        memory_allocator.clone(),
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

    let defect_map_view = ImageView::new_default(self.defect_map.clone()).unwrap();

    let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
    let set = DescriptorSet::new(
        descriptor_set_allocator,
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
        
    let dispatch_size_x = (image_width + local_size_x - 1) / local_size_x; 
    let dispatch_size_y = (image_height + local_size_y - 1) / local_size_y; 

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

        mod gain_correction_shader {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450

                    layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

                    layout(set = 0, binding = 0, r16ui) uniform uimage2D tex;
                    layout(set = 0, binding = 1, r16ui) uniform uimage2D gain_map_tex;

                    void main() {
                        ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
                        vec4 val = imageLoad(tex, coords);
                        vec4 dark_map_val = imageLoad(gain_map_tex, coords);
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

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    )); 

    let image_upload_buffer = Buffer::from_iter(
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
        // Iterator that produces the data.
        vec![0u16; (image_height*image_width) as usize],
    )
    .unwrap();


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
        vec![0u16; 4800*5800] /* number of elements, matching the image size */,
    )
    .unwrap();

    let defect_resources = DefectMapResources::new(device.clone(), queue.clone(), command_buffer_allocator.clone(), image_height as u32, image_width as u32);

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
            usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE , ..Default::default()
        }
    ).unwrap();

    builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(image_upload_buffer.clone(), image.clone())).unwrap();

    let image_view: Arc<ImageView> = ImageView::new_default(image.clone()).unwrap();

    {
        defect_resources.apply_defect_pipeline(&mut builder, device.clone(), queue.clone(), image_width as u32, image_height as u32, image_view.clone());
    }

    /*
    builder
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image.clone(), readback_buffer.clone()))
        .unwrap();
    */

    let command_buffer = builder.end().unwrap();
    let time = std::time::Instant::now();

    let future = sync::now(device)
    .then_execute(queue, command_buffer)
    .unwrap()
    .then_signal_fence_and_flush()
    .unwrap();

    future.wait(None).unwrap();
    println!("{:?}", time.elapsed());

    println!("Success");
}
