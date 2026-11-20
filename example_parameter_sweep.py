"""
示例4: 参数扫描

演示如何批量扫描不同pinhole直径对光斑分布的影响。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from xrf_sim import (
    XRFSystemConfig,
    AnnularSourceConfig,
    BeWindowConfig,
    PinholeConfig,
    WaferPlaneConfig,
    SimulationConfig,
    SpotSimulator,
)


def main():
    # 扫描不同pinhole直径
    pinhole_diameters = [1.5, 3.0, 5.0, 10.0, 20.0, 50.0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    results_summary = []

    for i, ph_diam in enumerate(pinhole_diameters):
        print(f"\n--- Pinhole直径 = {ph_diam} mm ---")

        config = XRFSystemConfig(
            source=AnnularSourceConfig(inner_radius=3.6, outer_radius=8.5, z_position=28.0),
            be_window=BeWindowConfig(diameter=11.2),
            pinhole=PinholeConfig(diameter=ph_diam, z_position=12.8),
            wafer=WaferPlaneConfig(z_position=0.0, size=70.0, resolution=350),
            simulation=SimulationConfig(num_rays=500_000, progress_bar=False),
        )

        sim = SpotSimulator(config)
        results = sim.run()

        # 绘制
        image = results['detector_image']
        size = config.wafer.size
        extent = [-size/2, size/2, -size/2, size/2]

        im = axes[i].imshow(image, extent=extent, cmap='jet', origin='lower')
        axes[i].set_title(f'Pinhole = {ph_diam} mm\n'
                          f'WD: {results["rays_on_wd_area"]:,} | '
                          f'ED: {results["rays_on_ed_area"]:,}')
        axes[i].set_xlabel('X (mm)')
        axes[i].set_ylabel('Y (mm)')
        plt.colorbar(im, ax=axes[i])

        results_summary.append({
            'pinhole_diameter': ph_diam,
            'rays_on_wd': results['rays_on_wd_area'],
            'rays_on_ed': results['rays_on_ed_area'],
            'efficiency': results['efficiency'],
        })

    fig.suptitle('Pinhole Diameter Sweep', fontsize=16)
    fig.tight_layout()
    fig.savefig('results/pinhole_sweep.png', dpi=150, bbox_inches='tight')
    print("\n已保存: results/pinhole_sweep.png")

    # 打印汇总
    print("\n" + "=" * 60)
    print(f"{'Pinhole(mm)':>12} {'WD rays':>12} {'ED rays':>12} {'Efficiency':>12}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['pinhole_diameter']:>12.1f} {r['rays_on_wd']:>12,} "
              f"{r['rays_on_ed']:>12,} {r['efficiency']:>11.4f}%")

    plt.close('all')


if __name__ == "__main__":
    main()
