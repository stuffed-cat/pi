use std::io;
use std::sync::Arc;
use std::thread;

use wgpu::util::DeviceExt;

fn main() {
    println!("蒙特卡洛方法计算π");
    println!("请选择计算方式:");
    println!("1. CPU多线程计算");
    println!("2. GPU计算");
    
    let mut choice = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("读取选择失败");
    
    let use_gpu = match choice.trim() {
        "2" => true,
        _ => false,
    };
    
    println!("请输入采样点数量：");
    
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("读取输入失败");
    
    let n: u64 = match input.trim().parse() {
        Ok(num) => {
            if num == 0 {
                println!("采样点数量必须大于0，使用默认值 1000000");
                1000000
            } else if num > 1_000_000_000_000 {
                println!("采样点数量太大，限制为最大值 1,000,000,000,000");
                1_000_000_000_000
            } else {
                num
            }
        }
        Err(_) => {
            println!("输入无效，使用默认值 1000000");
            1000000
        }
    };
    
    println!("开始计算...");
    let start_time = std::time::Instant::now();
    
    let pi = if use_gpu {
        println!("使用GPU计算");
        pollster::block_on(calculate_pi_gpu(n))
    } else {
        println!("使用CPU多线程计算");
        calculate_pi_monte_carlo_mt(n)
    };
    
    let duration = start_time.elapsed();
    
    println!("使用 {} 个采样点", n);
    println!("计算得到的π值: {}", pi);
    println!("真实π值: {}", std::f64::consts::PI);
    println!("误差: {}", (pi - std::f64::consts::PI).abs());
    println!("计算耗时: {:.2?}", duration);
}

fn calculate_pi_monte_carlo_mt(n: u64) -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    // 获取系统CPU核心数
    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4) as u64;
    
    println!("使用 {} 个线程", num_threads);
    
    // 使用原子类型来安全地累计结果
    let inside_circle = Arc::new(AtomicU64::new(0));
    
    // 计算每个线程处理的采样点数
    let samples_per_thread = n / num_threads;
    let remainder = n % num_threads;
    
    let mut handles = Vec::new();
    
    for i in 0..num_threads {
        let inside_circle_clone = Arc::clone(&inside_circle);
        
        // 为最后一个线程分配剩余的采样点
        let samples = if i == num_threads - 1 {
            samples_per_thread + remainder
        } else {
            samples_per_thread
        };
        
        let handle = thread::spawn(move || {
            use std::time::{SystemTime, UNIX_EPOCH};
            
            // 每个线程使用不同的随机种子
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64 + i * 1000000;
            
            let mut rng = SimpleRng::new(seed);
            let mut local_inside = 0;
            
            for _ in 0..samples {
                // 生成 [-1, 1] 范围内的随机点
                let x = rng.next_f64() * 2.0 - 1.0;
                let y = rng.next_f64() * 2.0 - 1.0;
                
                // 检查点是否在单位圆内
                if x * x + y * y <= 1.0 {
                    local_inside += 1;
                }
            }
            
            // 原子地累加到全局计数器
            inside_circle_clone.fetch_add(local_inside, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    // 等待所有线程完成
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_inside = inside_circle.load(Ordering::Relaxed);
    
    // π = 4 * (圆内点数 / 总点数)
    4.0 * total_inside as f64 / n as f64
}

// 简单的线性同余生成器
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }
    
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    
    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}

// GPU计算相关结构
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInput {
    samples_per_workgroup: u32,
    extra_samples: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOutput {
    inside_count: u32,
}

async fn calculate_pi_gpu(n: u64) -> f64 {
    calculate_pi_gpu_impl(n).await
}

async fn calculate_pi_gpu_impl(n: u64) -> f64 {
    // 初始化GPU设备
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        )
        .await
        .unwrap();

    println!("GPU设备: {}", adapter.get_info().name);
    
    // 获取GPU的真实限制
    let limits = device.limits();
    let max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
    let max_buffer_size = limits.max_buffer_size;
    
    println!("GPU限制: 最大工作组数 = {}, 最大缓冲区大小 = {} MB", 
             max_compute_workgroups_per_dimension, 
             max_buffer_size / 1024 / 1024);
    
    // 智能计算工作组数量 - 进一步降低每组样本数
    let max_samples_per_workgroup = 50000u32; // 每组最多5万样本，确保GPU稳定性
    
    // 计算最优工作组配置
    let ideal_workgroups = std::cmp::min(
        max_compute_workgroups_per_dimension,
        ((n as f64) / (max_samples_per_workgroup as f64)).ceil() as u32
    );
    
    // 如果需要的工作组数超过GPU限制，调整每组样本数
    let actual_workgroups = if ideal_workgroups > max_compute_workgroups_per_dimension {
        max_compute_workgroups_per_dimension
    } else {
        ideal_workgroups
    };
    
    // 平均分配样本到工作组
    let samples_per_workgroup = (n / actual_workgroups as u64) as u32;
    let extra_samples = (n % actual_workgroups as u64) as u32;
    
    // 检查每组样本数是否仍然过多
    if samples_per_workgroup > 1000000 {
        println!("警告: 每个工作组仍需处理 {} 个样本，可能导致GPU超时", samples_per_workgroup);
        println!("建议减少采样点数量或使用CPU计算");
    }
    
    println!("GPU配置: 工作组数 = {}, 每组基础样本数 = {}, 额外样本 = {}", 
             actual_workgroups, samples_per_workgroup, extra_samples);
    println!("GPU将处理全部 {} 个采样点", n);
    
    // 创建计算着色器 - 修复随机数生成和样本数分配问题
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Monte Carlo Pi Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(r#"
struct Input {
    samples_per_workgroup: u32,
    extra_samples: u32,
}

struct Output {
    inside_count: u32,
}

@group(0) @binding(0)
var<uniform> input: Input;

@group(0) @binding(1)
var<storage, read_write> output: array<Output>;

// 改进的随机数生成器 - 使用xorshift32算法
fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return x;
}

fn random_f32(state: ptr<function, u32>) -> f32 {
    return f32(xorshift32(state)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_id = global_id.x;
    if (workgroup_id >= arrayLength(&output)) {
        return;
    }
    
    // 确保每个工作组有不同的随机种子
    var rng_state = 12345u + workgroup_id * 1013904223u + 1u;
    if (rng_state == 0u) {
        rng_state = 1u;
    }
    
    var inside_count = 0u;
    
    // 基础样本数
    var samples_to_process = input.samples_per_workgroup;
    
    // 前extra_samples个工作组需要处理额外的一个样本
    if (workgroup_id < input.extra_samples) {
        samples_to_process = samples_to_process + 1u;
    }
    
    // 处理分配给这个工作组的所有样本
    for (var i = 0u; i < samples_to_process; i++) {
        let x = random_f32(&rng_state) * 2.0 - 1.0;
        let y = random_f32(&rng_state) * 2.0 - 1.0;
        
        let distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            inside_count = inside_count + 1u;
        }
    }
    
    // 将结果写入输出缓冲区
    output[workgroup_id].inside_count = inside_count;
}
        "#.into()),
    });

    // 创建输入缓冲区
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[GpuInput {
            samples_per_workgroup: samples_per_workgroup,
            extra_samples: extra_samples,
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // 创建输出缓冲区
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (actual_workgroups * std::mem::size_of::<GpuOutput>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 创建读取缓冲区
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建绑定组布局
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建绑定组
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // 创建计算管线
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "main",
    });

    // 执行计算
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(actual_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
    queue.submit(std::iter::once(encoder.finish()));

    // 读取结果
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: &[GpuOutput] = bytemuck::cast_slice(&data);
        
        let total_inside: u64 = result.iter().map(|r| r.inside_count as u64).sum();
        
        println!("GPU计算完成: 圆内点数 = {}, 总样本数 = {} (100%GPU处理)", total_inside, n);
        
        // 检查结果是否合理
        if total_inside == 0 && n > 0 {
            println!("错误: GPU计算返回0个圆内点数，可能是计算错误");
            println!("前几个工作组的结果: {:?}", &result[..std::cmp::min(5, result.len())]);
            println!("回退到CPU计算");
            drop(data);
            staging_buffer.unmap();
            return calculate_pi_monte_carlo_mt(n);
        }
        
        let pi_estimate = 4.0 * total_inside as f64 / n as f64;
        if pi_estimate < 1.0 || pi_estimate > 5.0 {
            println!("警告: GPU计算结果异常 (π ≈ {:.6}), 可能有错误", pi_estimate);
        }
        
        drop(data);
        staging_buffer.unmap();
        
        // 纯GPU计算结果
        pi_estimate
    } else {
        panic!("GPU计算失败");
    }
}
