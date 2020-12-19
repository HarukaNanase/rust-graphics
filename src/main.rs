use backend::Backend;
use renderer::{Renderer};

mod renderer;


fn main(){

    use gfx_hal::{
        window::{Extent2D},
    };

    let window_size : [u32; 2] = [800,600];    
    let event_loop = winit::event_loop::EventLoop::new();

    let(logical_window_size, physical_window_size) = {
        use winit::dpi::{LogicalSize, PhysicalSize};


        let dpi = event_loop.primary_monitor().unwrap().scale_factor();
        let logical: LogicalSize<u32> = LogicalSize{ width: window_size[0], height: window_size[1]};
        let physical: PhysicalSize<u32> = logical.to_physical(dpi);

        (logical, physical)

    };

    let surface_extent = Extent2D {
        width: physical_window_size.width,
        height: physical_window_size.height,
    };

    let window = winit::window::WindowBuilder::new().with_title("rust-graphics 3D 1.0")
    .with_inner_size(logical_window_size)
    .build(&event_loop)
    .expect("winit:: Failed to create a window.");


    let mut renderer = Renderer::<Backend>::new(&"rust-graphics", window, surface_extent);

    renderer.create_teapot();
    renderer.create_teapot();
    let start_time = std::time::Instant::now();




    //let mut should_configure_swapchain = true;
    //let renderer = Renderer::<Backend>::new(window, surface_extent);

    event_loop.run(move |event, _, control_flow| {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    renderer.set_new_surface_extent(dims);
                    renderer.set_should_configure_swapchain(true);
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    renderer.set_new_surface_extent(*new_inner_size);
                    renderer.set_should_configure_swapchain(true);
                }
                _ => (),
            },
            Event::MainEventsCleared => renderer.request_redraw(),
            Event::RedrawRequested(_) => renderer.draw(start_time),
            _ => (),
        }
    });

    
}


