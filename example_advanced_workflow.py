"""
示例5: 高级工作流 - 光斑导出、采样、数据交换

演示如何：
1. 模拟光斑分布
2. 导出为标准格式
3. 从该分布采样发射位置
4. 用于后续模拟或其他工具

这是与其他软件（如GEANT4、Monte Carlo N-Particle等）集成的典型工作流。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from xrf_sim import (
    XRFSystemConfig,
    AnnularSourceConfig,
    BeWindowConfig,
    PinholeConfig,
    WaferPlaneConfig,
    SimulationConfig,
    SpotSimulator,
    load_spot_profile,
    sample_from_profile,
    save_spot_profile,
)


def example1_basic_export():
    """
    示例1: 基础导出工作流
    
    输出标准二维强度数组（.npz）供其他程序使用
    """
    print("\n" + "=" * 60)
    print("示例1: 光斑分布导出")
    print("=" * 60)
    
    # 运行光斑模拟
    config = XRFSystemConfig(
        source=AnnularSourceConfig(inner_radius=3.6, outer_radius=8.5, z_position=28.0),
        be_window=BeWindowConfig(diameter=7.6),
        pinhole=PinholeConfig(diameter=3.09, z_position=13.1),
        wafer=WaferPlaneConfig(z_position=0.0, size=70.0, resolution=700),
        simulation=SimulationConfig(num_rays=50_000_000, progress_bar=False),
    )
    
    sim = SpotSimulator(config)
    results = sim.run()
    
    # 方式1: 直接从结果字典取得
    profile, x_range, y_range = results["spot_profile"]
    print(f"\n✅ 从结果取得光斑分布:")
    print(f"   形状: {profile.shape} (Ny, Nx)")
    print(f"   X范围: {x_range} mm")
    print(f"   Y范围: {y_range} mm")
    print(f"   强度总和: {profile.sum():.0f} 光子")
    
    # 方式2: 从模拟器对象取得
    profile2, x_range2, y_range2 = sim.get_spot_profile()
    print(f"\n✅ 从模拟器直接取得:")
    print(f"   数据相同: {np.array_equal(profile, profile2)}")
    
    # 方式3: 导出为标准格式（最推荐）
    npz_path = sim.export_spot_profile("results/example5_wafer_profile.npz")
    print(f"\n✅ 导出为.npz标准格式:")
    print(f"   文件: {npz_path}")
    print(f"   包含: profile, x_range, y_range, coordinate_unit")
    
    # 验证可以重新加载
    loaded_profile, loaded_x, loaded_y = load_spot_profile(npz_path)
    print(f"\n✅ 验证重新加载:")
    print(f"   数据一致: {np.array_equal(profile, loaded_profile)}")
    print(f"   坐标一致: {loaded_x == x_range and loaded_y == y_range}")


def example2_sampling():
    """
    示例2: 从光斑分布采样发射位置
    
    用于后续探测器模拟、蒙特卡洛计算等
    """
    print("\n" + "=" * 60)
    print("示例2: 从光斑分布采样")
    print("=" * 60)
    
    # 加载光斑
    profile, x_range, y_range = load_spot_profile("results/example5_wafer_profile.npz")
    
    # 从该分布抽样不同数量的位置
    sample_sizes = [100, 1_000, 100_000]
    
    for n_samples in sample_sizes:
        positions = sample_from_profile(
            n_samples,
            profile,
            x_range,
            y_range,
            rng=np.random.default_rng(42),  # 固定随机种子以保证复现性
        )
        
        print(f"\n✅ 采样 {n_samples:,} 个位置:")
        print(f"   形状: {positions.shape}")
        print(f"   X范围: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] mm")
        print(f"   Y范围: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] mm")
        
        # 绘制采样分布
        if n_samples <= 1000:
            plt.figure(figsize=(8, 8))
            plt.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5)
            plt.xlim(x_range)
            plt.ylim(y_range)
            plt.xlabel("X (mm)")
            plt.ylabel("Y (mm)")
            plt.title(f"采样分布 (n={n_samples})")
            plt.gca().set_aspect('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"results/example5_sampled_positions_{n_samples}.png", dpi=150)
            plt.close()
            print(f"   已保存: results/example5_sampled_positions_{n_samples}.png")
    
    # 保存大量采样结果供外部使用
    large_sample = sample_from_profile(1_000_000, profile, x_range, y_range)
    np.savetxt(
        "results/example5_1M_sample_positions.txt",
        large_sample,
        fmt="%.6f",
        header="X(mm) Y(mm)",
        comments=""
    )
    print(f"\n✅ 导出 100万 个采样位置:")
    print(f"   文件: results/example5_1M_sample_positions.txt")
    print(f"   格式: 纯文本, 2列 (X, Y)")


def example3_compare_distributions():
    """
    示例3: 比较不同参数下的光斑分布
    
    快速参数扫描工作流
    """
    print("\n" + "=" * 60)
    print("示例3: 参数扫描与对比")
    print("=" * 60)
    
    pinhole_sizes = [1.5, 3.09, 5.0, 10.0]
    results_list = []
    
    for ph_size in pinhole_sizes:
        print(f"\n运行 Pinhole = {ph_size} mm...")
        
        config = XRFSystemConfig(
            source=AnnularSourceConfig(inner_radius=3.6, outer_radius=8.5, z_position=28.0),
            be_window=BeWindowConfig(diameter=7.6),
            pinhole=PinholeConfig(diameter=ph_size, z_position=13.1),
            wafer=WaferPlaneConfig(z_position=0.0, size=70.0, resolution=350),
            simulation=SimulationConfig(num_rays=10_000_000, progress_bar=False),
        )
        
        sim = SpotSimulator(config)
        results = sim.run()
        
        profile, x_range, y_range = results["spot_profile"]
        
        results_list.append({
            'pinhole': ph_size,
            'profile': profile,
            'x_range': x_range,
            'y_range': y_range,
            'intensity_sum': profile.sum(),
            'intensity_max': profile.max(),
            'wd_rays': results['rays_on_wd_area'],
            'ed_rays': results['rays_on_ed_area'],
        })
    
    # 绘制对比
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, res in enumerate(results_list):
        ax = axes.flatten()[idx]
        
        im = ax.imshow(
            res['profile'],
            extent=[res['x_range'][0], res['x_range'][1],
                   res['y_range'][0], res['y_range'][1]],
            origin='lower',
            cmap='jet'
        )
        
        ax.set_title(f"Pinhole = {res['pinhole']} mm\n"
                    f"WD:{res['wd_rays']:,} | ED:{res['ed_rays']:,}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        plt.colorbar(im, ax=ax)
    
    fig.suptitle("Pinhole直径扫描对比", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/example5_pinhole_sweep_compare.png", dpi=150)
    plt.close()
    print(f"\n✅ 对比图已保存: results/example5_pinhole_sweep_compare.png")
    
    # 打印统计表
    print(f"\n{'Pinhole(mm)':>12} {'强度总和':>12} {'强度峰值':>12} {'WD光线':>12} {'ED光线':>12}")
    print("-" * 60)
    for res in results_list:
        print(f"{res['pinhole']:>12.2f} {res['intensity_sum']:>12.0f} "
              f"{res['intensity_max']:>12.0f} {res['wd_rays']:>12,} {res['ed_rays']:>12,}")


def example4_custom_profile():
    """
    示例4: 使用外部数据源的强度分布
    
    演示如何导入其他工具的结果并转换为标准格式
    """
    print("\n" + "=" * 60)
    print("示例4: 导入外部强度分布")
    print("=" * 60)
    
    # 生成一个人工强度分布（模拟来自其他工具的结果）
    res = 200
    x = np.linspace(-20, 20, res)
    y = np.linspace(-20, 20, res)
    X, Y = np.meshgrid(x, y)
    
    # 高斯光斑
    external_profile = np.exp(-0.5 * (X**2 + Y**2) / (5**2))
    external_profile *= 1e6  # 缩放到合理的计数范围
    
    x_range = (-20.0, 20.0)
    y_range = (-20.0, 20.0)
    
    print(f"\n✅ 生成人工强度分布:")
    print(f"   形状: {external_profile.shape}")
    print(f"   强度范围: [{external_profile.min():.0f}, {external_profile.max():.0f}]")
    
    # 转换为标准格式
    saved_path = save_spot_profile(
        "results/example5_external_profile.npz",
        external_profile,
        x_range,
        y_range,
    )
    print(f"\n✅ 保存为标准格式:")
    print(f"   文件: {saved_path}")
    
    # 验证并采样
    loaded, x_r, y_r = load_spot_profile(saved_path)
    samples = sample_from_profile(100_000, loaded, x_r, y_r)
    
    print(f"\n✅ 从导入的分布采样:")
    print(f"   采样点数: {len(samples):,}")
    print(f"   采样范围: X=[{samples[:, 0].min():.2f}, {samples[:, 0].max():.2f}] mm")
    
    # 绘制对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = ax1.imshow(external_profile, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     origin='lower', cmap='jet')
    ax1.set_title("原始外部分布")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    plt.colorbar(im1, ax=ax1)
    
    # 采样分布热图
    hist, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1],
                                   bins=[50, 50],
                                   range=[[x_range[0], x_range[1]], [y_range[0], y_range[1]]])
    im2 = ax2.imshow(hist.T, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                     origin='lower', cmap='jet')
    ax2.set_title("采样重建分布 (100K点)")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    plt.colorbar(im2, ax=ax2)
    
    fig.tight_layout()
    fig.savefig("results/example5_external_profile_compare.png", dpi=150)
    plt.close()
    print(f"   对比图已保存: results/example5_external_profile_compare.png")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # 运行所有示例
    example1_basic_export()
    example2_sampling()
    example3_compare_distributions()
    example4_custom_profile()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例完成！")
    print("=" * 60)
    print("\n生成的文件:")
    import glob
    for f in sorted(glob.glob("results/example5_*")):
        print(f"  {f}")
