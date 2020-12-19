#![allow(dead_code)]
#![allow(unused_variables)]
use std::mem::ManuallyDrop;
use serde::{Deserialize};
use cgmath::{Matrix3, Matrix4};
use shaderc::{CompileOptions, ShaderKind};

use gfx_hal::{Backend, Instance, adapter::Adapter, device::Device, format::Format, pso::{AttributeDesc, Element, VertexBufferDesc}, window::{Extent2D, PresentationSurface, Surface}};
use gfx_hal::pool::{CommandPool, CommandPoolCreateFlags};
use gfx_hal::command::CommandBuffer;
use gfx_hal::queue::{QueueFamily, QueueGroup};
use winit::window::Window;
use std::fs;





#[derive(serde::Deserialize)]
#[repr(C)]
struct Vertex{
    position: [f32; 3],
    normal: [f32; 3]
}


struct Mesh{
    mesh: Vec<Vertex>,
    len: u32,
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Matrices{
    transform: Matrix4<f32>,
}

struct DrawableMesh{
    mesh: Mesh,
    matrices: Matrices,
}







pub const WINDOW_SIZE: [u32; 2] = [512,512];
pub struct Renderer<B: gfx_hal::Backend>{
    window : Window,
    instance: B::Instance,
    device : B::Device,
    //gpu : gfx_hal::adapter::Gpu<B>,
    adapter: Adapter<B>,
    queue_families: Vec<B::QueueFamily>,
    queue_groups: Vec<QueueGroup<B>>,
    viewport: gfx_hal::pso::Viewport,
    surface :B::Surface,
    surface_extent:Extent2D,
    surface_color_format: gfx_hal::format::Format,
    render_passes : Vec<B::RenderPass>,
    pipeline_layouts: Vec<B::PipelineLayout>,
    pipelines : Vec<B::GraphicsPipeline>,
    //compiler: shaderc::Compiler,
    window_size: [u32; 2],
    window_title: String,
    command_pools: Vec<B::CommandPool>,
    command_buffers: Vec<B::CommandBuffer>,
    render_semaphores: Vec<B::Semaphore>,
    command_fences: Vec<B::Fence>,
    frame: u64,
    sampler: Option<ManuallyDrop<B::Sampler>>,
    should_configure_swapchain: bool,
    render_time_out: u64,
    meshes: Vec<Mesh>,
    memory: B::Memory,
    memory_buffer: B::Buffer,
}



impl<B: gfx_hal::Backend> Renderer<B>{


    pub fn new(window_title: &str, window: Window, surface_extent:Extent2D) -> Renderer<B>{

    let compiler = shaderc::Compiler::new().unwrap();
    let instance = Renderer::<B>::create_instance(window_title);
    let surface = Renderer::<B>::create_surface(&window, &instance); 
    let viewport = Renderer::<B>::create_viewport(0,0, surface_extent.width as i16, surface_extent.height as i16);
    let adapter = Renderer::<B>::create_adapter(&instance);
    let mut gpu = Renderer::<B>::create_gpu(&adapter, &surface);
    let device = gpu.device;
    let queue_group = gpu.queue_groups.pop().unwrap();
    let (command_pool, command_buffer) = Renderer::create_command_pool_and_buffer(&device, &queue_group, gfx_hal::command::Level::Primary);
    let surface_format = Renderer::create_surface_format(&surface, &adapter);
    let render_pass = Renderer::<B>::create_render_pass(&device, surface_format);

    unsafe{
        let pipeline_layout = Renderer::<B>::create_pipeline_layout(&device);
        let mut compiler = shaderc::Compiler::new().unwrap();
        let pipeline = Renderer::<B>::create_pipeline(&mut compiler, &device, &pipeline_layout, &render_pass, &".//assets//shaders//vertex.vert", &".//assets//shaders//fragment.frag");




        let(buf_mem, buffer) = {
            use gfx_hal::buffer::Usage;
            use gfx_hal::memory::Properties;

            Renderer::<B>::make_buffer(&device, &adapter.physical_device, 500000 * std::mem::size_of::<Vertex>(), Usage::VERTEX, Properties::CPU_VISIBLE)
        };
        let fence = Renderer::<B>::create_device_fence(&device);
        let semaphore = Renderer::<B>::create_device_semaphore(&device);

        let renderer = Renderer{
            window : window,
            instance: instance,
            device : device,
            adapter: adapter,
            queue_families: vec![],
            queue_groups: vec![queue_group],
            viewport: viewport,
            surface : surface,
            surface_extent: surface_extent,
            surface_color_format: surface_format,
            render_passes : vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines :vec![pipeline],
            window_size: [800,600],
            window_title: window_title.to_string(),
            command_pools: vec![command_pool],
            command_buffers: vec![command_buffer],
            render_semaphores: vec![semaphore],
            command_fences: vec![fence],
            frame: 0,
            sampler: None,
            should_configure_swapchain: true,
            render_time_out: 1_000_000_000,
            meshes: vec![],
            memory: buf_mem,
            memory_buffer: buffer ,
            };


            renderer

        }
    }


    fn create_instance(window_title: &str) -> B::Instance{
        
        let instance = B::Instance::create(window_title, 1).expect("Backend not supported");
        return instance;
    }

    fn create_surface(window: &Window, instance: &B::Instance) ->B::Surface {
        let surface = unsafe {
            instance.create_surface(window).expect("Failed to create surface in renderer.")
        };
        return surface;
    }

    fn create_adapter(inst: &B::Instance) -> Adapter<B>{
        let adapter = inst.enumerate_adapters().remove(0);
        return adapter;
    }

    fn create_gpu(adapter: &Adapter<B>, surface: &B::Surface) -> gfx_hal::adapter::Gpu<B>{
        let queue_family = adapter.queue_families.iter()
                        .find(|family| {
                            surface.supports_queue_family(family) && family.queue_type().supports_graphics()
                        }).expect("No compatible queue");

        let gpu = unsafe {
            use gfx_hal::adapter::PhysicalDevice;
            adapter.physical_device.open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
            .expect("Failed to open GPU device.")
        };


        gpu
    }

    fn create_viewport(x: i16, y: i16, width: i16, height: i16) -> gfx_hal::pso::Viewport{
        use gfx_hal::pso::{Rect, Viewport};
        
        
        Viewport{
            rect: Rect{
                x:x, y:y, w: width, h: height
            },
            depth: 0.0 .. 1.0,
        }        
    }

    fn create_command_pool_and_buffer(device: &B::Device, queue_group: &QueueGroup<B>, level: gfx_hal::command::Level) -> (B::CommandPool, B::CommandBuffer) {

        unsafe{

            let mut command_pool = device.create_command_pool(queue_group.family, CommandPoolCreateFlags::empty()).expect("Out of memory for pools");

            let command_buffer = command_pool.allocate_one(level);

            (command_pool, command_buffer)
        }
    }

    fn create_device_fence(device: &B::Device) -> B::Fence{
        device.create_fence(true).expect("Failed to create GPU fence.")
    }

    fn create_device_semaphore(device: &B::Device) -> B::Semaphore{
        device.create_semaphore().expect("Failed to create GPU semaphore.")
    }

    fn create_surface_format(surface: &B::Surface, adapter: &Adapter<B>) -> gfx_hal::format::Format {
        use gfx_hal::format::{ChannelType, Format};
        let supported_formats = surface.supported_formats(&adapter.physical_device).unwrap_or(vec![]);

        let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

        let format = supported_formats.into_iter().find(|format| format.base_format().1 == ChannelType::Srgb).unwrap_or(default_format);
        
        format
    }

    fn create_render_pass(device: &B::Device, surface_color_format: gfx_hal::format::Format) -> B::RenderPass {
        use gfx_hal::image::Layout;
        use gfx_hal::pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc};

        let color_attachment = Attachment{
            format: Some(surface_color_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::Present,

        };

        let subpass = SubpassDesc{
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe{
            device.create_render_pass(&[color_attachment], &[subpass], &[]).expect("Out of memory to create a render pass.")
        }
    }


    unsafe fn create_pipeline_layout(device: &B::Device) -> B::PipelineLayout {
        use gfx_hal::pso::ShaderStageFlags;
        let push_constant_bytes = std::mem::size_of::<PushConstants>() as u32;
        device.create_pipeline_layout(&[], &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)]).expect("Failed to create pipeline layout.")   
    }

    unsafe fn create_pipeline(shader_compiler: &mut shaderc::Compiler, device: &B::Device, pipeline_layout: &B::PipelineLayout, 
                                render_pass: &B::RenderPass, vertex_shader: &str, fragment_shader: &str) 
    -> B::GraphicsPipeline
    {        
        let vertex_shader = fs::read_to_string(vertex_shader).expect("Failed to open/read Vertex Shader supplied.");
        let fragment_shader = fs::read_to_string(fragment_shader).expect("Failed to open/read Fragment Shader supplied: {}");

        Renderer::<B>::make_pipeline(shader_compiler, device, render_pass, pipeline_layout, &vertex_shader, &fragment_shader)
    }

    unsafe fn make_pipeline(
        shader_compiler: &mut shaderc::Compiler,
        device: &B::Device,
        render_pass: &B::RenderPass,
        pipeline_layout: &B::PipelineLayout,
        vertex_shader: &str,
        fragment_shader: &str,
    ) -> B::GraphicsPipeline {
        use gfx_hal::{
            pass::Subpass,
            pso::{ BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face, GraphicsPipelineDesc, InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer, Specialization},
        };

        let compiled_vert_shader = Renderer::<B>::compile_shader(shader_compiler, vertex_shader, ShaderKind::Vertex, "unnamed", "main", Option::None);
        let compiled_frag_shader = Renderer::<B>::compile_shader(shader_compiler, fragment_shader, ShaderKind::Fragment, "unnamed", "main", Option::None);

        let vertex_shader_module = device.create_shader_module(&compiled_vert_shader).unwrap();
        let frag_shader_module = device.create_shader_module(&compiled_frag_shader).unwrap();


        let(vs_entry, fs_entry) : (EntryPoint<B>, EntryPoint<B>) = (
            EntryPoint{entry: "main", module: &vertex_shader_module, specialization: Specialization::default()},
            EntryPoint{entry: "main", module: &frag_shader_module, specialization: Specialization::default()},
        );

        let primitive_assembler = PrimitiveAssemblerDesc::Vertex{
            buffers: &[VertexBufferDesc{
                binding:0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: gfx_hal::pso::VertexInputRate::Vertex,
            }],
            attributes: &[
                AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: Element{
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    }
                },
                AttributeDesc{
                    location: 1,
                    binding:0,
                    element: Element{
                        format: Format::Rgb32Sfloat,
                        offset: 12,
                    }
                },
            ],
            input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
            vertex: vs_entry,
            tessellation: None,
            geometry: None,
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(primitive_assembler, Rasterizer{cull_face: Face::BACK, .. Rasterizer::FILL}
            , Some(fs_entry), pipeline_layout, Subpass{ index: 0, main_pass: render_pass});

        pipeline_desc.blender.targets.push(ColorBlendDesc{
            mask: ColorMask::ALL,
            blend: Some(BlendState::ALPHA),
        });
        
        let pipeline = device.create_graphics_pipeline(&pipeline_desc, None).unwrap();

        device.destroy_shader_module(vertex_shader_module);
        device.destroy_shader_module(frag_shader_module);

        pipeline

    }

    fn compile_shader(compiler: &mut shaderc::Compiler, glsl: &str, shader_kind: ShaderKind, input_file_name: &str, entry_point_name: &str, additional_options: Option<&CompileOptions>) -> Vec<u32>{
        
        let compiled_shader = compiler.compile_into_spirv(glsl, shader_kind, input_file_name, entry_point_name, additional_options);
        
        return compiled_shader.unwrap().as_binary().to_vec()
    }


    fn reconfigure_swapchain(&mut self){
        use gfx_hal::window::SwapchainConfig;

            let capabilities = self.surface.capabilities(&self.adapter.physical_device);

            let mut swapchain_configuration = SwapchainConfig::from_caps(&capabilities, self.surface_color_format, self.surface_extent);

            if capabilities.image_count.contains(&3){
                swapchain_configuration.image_count = 3;
            }

            self.surface_extent = swapchain_configuration.extent;


            unsafe{
                self.surface.configure_swapchain(&self.device, swapchain_configuration).expect("Failed to configure swapchain");
                println!("Re-configured swapchain.")
            };

            self.viewport = gfx_hal::pso::Viewport{ rect: gfx_hal::pso::Rect{
                x:0, y:0, w: self.surface_extent.width as i16, h: self.surface_extent.height as i16,
            }, depth: 0.0 .. 1.0,
         };

            self.should_configure_swapchain = false;
    }


    fn create_anim(time_elapsed : std::time::Instant) -> Vec<PushConstants>{

        let angle = time_elapsed.elapsed().as_secs_f32();
        
        
        let teapots = &[
            PushConstants {
                transform: Renderer::<B>::make_transform([0.5, 0., 0.5], angle, 1.0),
            },
            PushConstants{
                transform: Renderer::<B>::make_transform([-0.5, 0., 0.5], angle, 1.0),
            }
        ];

        teapots.to_vec()
    }

    unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
        let size_in_bytes = std::mem::size_of::<T>();
        let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
        let start_ptr = push_constants as *const T as *const u32;
        std::slice::from_raw_parts(start_ptr, size_in_u32s)
    }


    pub fn draw(&mut self, time: std::time::Instant){

        
        if self.should_configure_swapchain{
            self.reconfigure_swapchain();
        }
        
        let  render_pass = &mut self.render_passes[0];
        let  pipeline = &mut self.pipelines[0];
        let  command_buffer = &mut self.command_buffers[0];
        let  pipeline_layout = &mut self.pipeline_layouts[0];

        unsafe{

            self.device.wait_for_fence(&self.command_fences[0], self.render_time_out as u64).expect("Failed to wait for fence. Device lost???");
            self.device.reset_fence(&self.command_fences[0]).expect("Failed to reset fence.");    
            self.command_pools[0].reset(false);
        }


        let surface_image = unsafe{
            match self.surface.acquire_image(self.render_time_out) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.should_configure_swapchain = true;
                    return;
                }
            }
        };

        let framebuffer = unsafe{
            use std::borrow::Borrow;

            use gfx_hal::image::Extent;

            self.device.create_framebuffer(
                render_pass, 
                vec![surface_image.borrow()], 
                Extent{
                    width: self.surface_extent.width,
                    height: self.surface_extent.height,
                    depth: 1,
                },
            ).unwrap()
        };

        unsafe{
            use gfx_hal::command::{ClearColor, ClearValue, CommandBufferFlags, SubpassContents};


            command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);
            command_buffer.set_viewports(0, &[self.viewport.clone()]);
            command_buffer.set_scissors(0, &[self.viewport.rect]);

            command_buffer.bind_vertex_buffers(
                0,
                vec![(&self.memory_buffer, gfx_hal::buffer::SubRange::WHOLE)],
            );

            command_buffer.begin_render_pass(
                render_pass,
                &framebuffer,
                self.viewport.rect,
                &[ClearValue {
                    color: ClearColor{ float32: [0.0, 0.0, 0.0, 1.0]}
                }],
                SubpassContents::Inline
            );



            command_buffer.bind_graphics_pipeline(pipeline);

            let teapots = &Renderer::<B>::create_anim(time);

            use gfx_hal::pso::ShaderStageFlags;
            
            let n_meshes = self.meshes.len() as i32;

            for i in 0..n_meshes{
              command_buffer.push_graphics_constants(
                    pipeline_layout,
                    ShaderStageFlags::VERTEX,
                    0,
                    Renderer::<B>::push_constant_bytes(&teapots[i as usize]),
                );

                command_buffer.draw(0..self.meshes[i as usize].len, 0..1);
            }

           

            command_buffer.end_render_pass();
            command_buffer.finish();
        }

        unsafe{
            use gfx_hal::queue::{CommandQueue, Submission};



            let submission = Submission{
                command_buffers: vec![&command_buffer],
                wait_semaphores: None,
                signal_semaphores: vec![&self.render_semaphores[0]]
            };


            let queue = &mut self.queue_groups[0].queues[0];
           // self.queue_groups[0].queues[0].submit(submission, Some(&self.command_fences[0]));
            queue.submit(submission, Some(&self.command_fences[0]));
            let result = queue.present(&mut self.surface, surface_image, Some(&self.render_semaphores[0]));

            self.should_configure_swapchain |= result.is_err();


            self.device.destroy_framebuffer(framebuffer);
        } 
    }


   

    pub fn set_new_surface_extent(&mut self, dims: winit::dpi::PhysicalSize<u32>) -> (){
        self.surface_extent = Extent2D{ width: dims.width, height: dims.height};
    }

    pub fn set_should_configure_swapchain(&mut self, new_value: bool){
        self.should_configure_swapchain = new_value;
    }


    pub fn request_redraw(&self){
        self.window.request_redraw();
    }



    fn create_teapot_vertices() -> Vec<Vertex>{
        let mesh_data = include_bytes!("..//assets//models//teapot_mesh.bin");

        let mesh: Vec<Vertex> = bincode::deserialize(mesh_data).expect("Failed to deserialize teapot mesh.");
        
        return mesh;

    }

    pub fn create_teapot(&mut self) -> (){
        let mesh = Renderer::<B>::create_teapot_vertices();
        let buffer_len = mesh.len() * std::mem::size_of::<Vertex>();
        let mesh_len = mesh.len();
        println!("Size of buffer needed: {}", buffer_len);

        unsafe {
            use gfx_hal::memory::Segment;
    
            let mapped_memory = self.device
                .map_memory(&self.memory, Segment::ALL)
                .expect("Failed to map memory");
    
            std::ptr::copy_nonoverlapping(mesh.as_ptr() as *const u8, mapped_memory, buffer_len);
    
            &self.device
                .flush_mapped_memory_ranges(vec![(&self.memory, Segment::ALL)])
                .expect("Out of memory");
    
            &self.device.unmap_memory(&self.memory);
        };

        &self.meshes.push(Mesh{
            mesh: mesh,
            len: mesh_len as u32,
        });

        println!("Size of meshes: {}", &self.meshes.len())
    }
    

    unsafe fn make_buffer(device: &B::Device, physical_device: &B::PhysicalDevice,
                                                buffer_len: usize, usage: gfx_hal::buffer::Usage, properties: gfx_hal::memory::Properties)
        -> (B::Memory, B::Buffer){

            use gfx_hal::{adapter::PhysicalDevice, MemoryTypeId};
            let mut buffer= device.create_buffer(buffer_len as u64, usage).expect("Failed to create buffer.");

            let requirements = device.get_buffer_requirements(&buffer);

            let mem_types = physical_device.memory_properties().memory_types;

            let mem_type = mem_types.iter().enumerate().find(|(id, mem_type)| {
                    let type_supported = requirements.type_mask & (1_u32 << id) != 0;
                    type_supported && mem_type.properties.contains(properties)
                }).map(|(id, _ty)| MemoryTypeId(id))
                .expect("No compatible memory type found.");
            

            let buf_mem = device.allocate_memory(mem_type, requirements.size).expect("Failed to allocate memory in the device.");
            device.bind_buffer_memory(&buf_mem, 0, &mut buffer).expect("Failed to bind memory to device.");

            (buf_mem, buffer)
        }

        fn make_transform(translate: [f32; 3], angle: f32, scale: f32) -> Matrix4<f32> {
            let c = angle.cos() * scale;
            let s = angle.sin() * scale;
            let [dx, dy, dz] = translate;

            Matrix4::new(
                c, 0., s, 0.,
                0., scale, 0., 0.,
                -s, 0., c, 0.,
                dx, dy, dz, 1.,
            )
                
            
        }

        unsafe fn free_memory(&self){
            
        }

}









pub struct BackendResources<B: gfx_hal::Backend> {
    pub instance: B::Instance,
    pub surface: B::Surface,
    pub device: B::Device,
    pub render_passes: Vec<B::RenderPass>,
    pub pipeline_layouts: Vec<B::PipelineLayout>,
    pub pipelines: Vec<B::GraphicsPipeline>,
    pub command_pool: B::CommandPool,
    pub submission_fence: B::Fence,
    pub rendering_semaphore: B::Semaphore,
}
pub struct BackendResourceHolder<B: Backend>(pub ManuallyDrop<BackendResources<B>>);
impl<B: Backend> Drop for BackendResourceHolder<B>{
    fn drop(&mut self){
        unsafe{
            let BackendResources {
                instance, mut surface, device, command_pool, render_passes, 
                pipeline_layouts, pipelines,submission_fence, rendering_semaphore
            } = ManuallyDrop::take(&mut self.0);
            
            device.destroy_semaphore(rendering_semaphore);
            device.destroy_fence(submission_fence);

            for pipeline in pipelines{
                device.destroy_graphics_pipeline(pipeline);
            }

            for pipeline_layout in pipeline_layouts{
                device.destroy_pipeline_layout(pipeline_layout);
            }

            for render_pass in render_passes{
                device.destroy_render_pass(render_pass);
            }

            device.destroy_command_pool(command_pool);
            surface.unconfigure_swapchain(&device);
            instance.destroy_surface(surface);
        }
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PushConstants {
    transform: Matrix4<f32>,
}