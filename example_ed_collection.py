"""
示例2: ED通道收集效率模拟

模拟EDXRF探测器的收集分布，计算wafer面上各位置被探测器收集的概率。
对应原始代码: EDcollection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from xrf_sim import (
    EDCollectionSystemConfig,
    EDSourceConfig,
    EDShutterConfig,
    EDDetectorConfig,
    EDOccluderConfig,
    SimulationConfig,
    EDCollectionSimulator,
    plot_ed_source_density,
    plot_ed_collection_probability,
    plot_ed_detector_image,
)


def main():
    # ========== 第1步: 配置系统参数 ==========
    config = EDCollectionSystemConfig(
        source=EDSourceConfig(
            width=60.0,
            height=60.0,
            z_position=0.0,
            tilt_angle_x=0.0,
            tilt_angle_y=-58.0,
            density_resolution=600,
        ),
        shutter=EDShutterConfig(
            semi_minor=2.5,        # Mo挡板短轴
            semi_major=2.0,        # Mo挡板长轴
            z_position=99.58,
        ),
        detector=EDDetectorConfig(
            z_position=169.08,
            shape='circular',
            diameter=7.7,
            resolution=200,
        ),
        occluder=EDOccluderConfig(
            diameter=20.75,
            distance_from_source=12.8,
        ),
        simulation=SimulationConfig(
            num_rays=1_000_000_00,    # 演示用，实际可设为1千万
            theta_range=(56.727, 59.273),  # 58 +/- 1.273 (soller slit)
            phi_range=(-10.0, 10.0),
            store_traces=True,
            max_traces=1000,
        ),
    )

    # ========== 第2步: 运行模拟 ==========
    sim = EDCollectionSimulator(config)
    results = sim.run()

    # ========== 第3步: 保存结果 ==========
    filepath = sim.save()

    # ========== 第4步: 可视化 ==========
    # 光源面密度分布
    fig1, _ = plot_ed_source_density(results)
    fig1.savefig('results/ed_source_density.png', dpi=150, bbox_inches='tight')
    print("已保存: results/ed_source_density.png")

    # 收集概率分布
    fig2, _ = plot_ed_collection_probability(results)
    fig2.savefig('results/ed_collection_prob.png', dpi=150, bbox_inches='tight')
    print("已保存: results/ed_collection_prob.png")

    # 探测器图像
    fig3, _ = plot_ed_detector_image(results)
    fig3.savefig('results/ed_detector_image.png', dpi=150, bbox_inches='tight')
    print("已保存: results/ed_detector_image.png")

    plt.close('all')
    print("\nED收集模拟完成!")


if __name__ == "__main__":
    main()
