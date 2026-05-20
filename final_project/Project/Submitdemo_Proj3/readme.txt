# =========================
# 关于提供的文件
# =========================
Deg文件夹内有10例3D ASL退化后的图像，
格式为nii.gz，可调用nibabel库函数读入


# =========================
# 实验目的
# =========================
综合运用噪声估计、滤波去噪、锐化等知识，
完成图像恢复（ps：示意图仅供参考，仅图像
处理算法很难达到这个效果）


# =========================
# 测试指标
# =========================
同学可通过Noise level, Sharpness, High-freq energy
等no-reference指标辅助观察恢复效果
退化前图像与退化模型𝐷𝑒𝑔参数不公布，可忽略或自行估计

实际测试指标为Noise level, Sharpness, High-freq energy  
Reblur error(‖𝐷𝑒𝑔(𝐼_𝑟𝑒𝑠𝑡𝑜𝑟𝑒 )−𝐼‖)及
恢复图像与退化前图像的ssim，共计5个指标的加权，
各指标权重较为均衡
测试结果为提供的10个数据加上不公布的5个数据的平均

# =========================
# 提交文件
# =========================
submissions/
│   ├── proj3_student_001/
│   │   │── run.py
│   │   │── run.ipynb(可选)
│   │   │── ...
│   │   └── Restore/

按proj3_队长姓名_学号命名
需要提交一份过程中无图像展示，
有测试接口的py文件便于测试
参考测试接口主函数如下：

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".nii.gz")]

    for fname in files:
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)

        vol, affine, header = load_nii(in_path)
        result = process_one(vol)
        save_nii(out_path, result, affine, header)

其中process_one是单个图像恢复处理函数，return恢复后的图像矩阵，函数名可随意替换
开发过程中使用的ipynb与debug记录可作为辅助材料一并提交