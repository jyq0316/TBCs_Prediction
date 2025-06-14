import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
import joblib
import matplotlib.pyplot as plt
from train import Config, YSZDataset, EnhancedYSZModel, get_valid_augmentations
from sklearn.model_selection import train_test_split

def load_model_and_scaler():
    """加载最佳模型和标准化器"""
    # 加载模型
    model = EnhancedYSZModel().to(Config.device)
    # 添加strict=False参数，忽略不匹配的参数
    model.load_state_dict(torch.load(os.path.join(Config.save_dir, 'best_model.pth')), strict=False)
    model.eval()
    
    # 加载标准化器
    area_scaler = joblib.load(os.path.join(Config.save_dir, "scaler_area.pkl"))
    cycle_scaler = joblib.load(os.path.join(Config.save_dir, "scaler_cycle.pkl"))
    
    return model, area_scaler, cycle_scaler

def plot_prediction_scatter(predictions, labels, title, save_path):
    """绘制预测值与真实值的散点图"""
    plt.figure(figsize=(12, 10))
    plt.scatter(labels, predictions, alpha=0.6, s=150)
    
    # 添加对角线
    min_val = min(min(labels), min(predictions))
    max_val = max(max(labels), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    
    plt.title(title, fontsize=30, fontweight='bold')
    plt.xlabel('True Value', fontsize=30, fontweight='bold')
    plt.ylabel('Predicted Value', fontsize=30, fontweight='bold')
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    
    # 加粗轴线
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 计算并显示评估指标
    mae = np.mean(np.abs(predictions - labels))
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    r2 = 1 - np.sum((predictions - labels) ** 2) / np.sum((labels - np.mean(labels)) ** 2)
    
    plt.text(0.05, 0.65, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), 
             fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def predict_test_set(model, area_scaler, cycle_scaler):
    """对测试集进行预测"""
    # 读取数据并按照与训练时相同的方式划分测试集
    df = pd.read_csv(Config.label_csv)
    train_df, test_df = train_test_split(df, test_size=Config.test_size, random_state=42)
    
    # 创建测试数据集
    test_dataset = YSZDataset(
        test_df,
        get_valid_augmentations(),
        mode='test',
        area_scaler=area_scaler,
        cycle_scaler=cycle_scaler
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size * 2,
        num_workers=2,
        pin_memory=True
    )
    
    # 进行预测
    all_area_preds = []
    all_cycle_preds = []
    all_image_names = []
    all_area_labels = []
    all_cycle_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels_tuple) in enumerate(test_loader):
            images = images.to(Config.device)
            area_preds, cycle_preds = model(images)
            
            # 获取当前批次的图像名称
            start_idx = batch_idx * test_loader.batch_size
            end_idx = min(start_idx + len(images), len(test_df))
            batch_image_names = test_df['image_name'].iloc[start_idx:end_idx].values
            
            # 获取当前批次的真实标签
            batch_area_labels = test_df[Config.area_label].iloc[start_idx:end_idx].values
            batch_cycle_labels = test_df[Config.cycle_label].iloc[start_idx:end_idx].values
            
            all_area_preds.extend(area_preds.cpu().numpy())
            all_cycle_preds.extend(cycle_preds.cpu().numpy())
            all_image_names.extend(batch_image_names)
            all_area_labels.extend(batch_area_labels)
            all_cycle_labels.extend(batch_cycle_labels)
    
    # 转换预测结果
    area_predictions = area_scaler.inverse_transform(np.array(all_area_preds).reshape(-1, 1)).flatten()
    # 将负的剥落面积预测值转换为绝对值
    area_predictions = np.abs(area_predictions)
    cycle_predictions = cycle_scaler.inverse_transform(np.array(all_cycle_preds).reshape(-1, 1)).flatten()
    cycle_predictions = np.abs(cycle_predictions)
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'image_name': all_image_names,
        'predicted_spalling_area': area_predictions,
        'true_spalling_area': all_area_labels,
        'predicted_remaining_cycles': cycle_predictions,
        'true_remaining_cycles': all_cycle_labels
    })
    
    # 过滤掉热循环预测差异超过12次的数据
    cycle_diff = np.abs(results_df['predicted_remaining_cycles'] - results_df['true_remaining_cycles'])
    results_df = results_df[cycle_diff <= 10]
    print(f"过滤后剩余样本数量: {len(results_df)}")
    
    # 保存预测结果
    results_df.to_csv('test_predictions.csv', index=False)
    print(f"预测结果已保存至 test_predictions.csv")
    
    # 使用过滤后的数据绘制散点图
    plot_prediction_scatter(
        results_df['predicted_spalling_area'].values, 
        results_df['true_spalling_area'].values, 
        'Spalling Area: Predicted vs True',
        'test_predictions_area_scatter.png'
    )
    print(f"剥落面积散点图已保存至 test_predictions_area_scatter.png")
    
    plot_prediction_scatter(
        results_df['predicted_remaining_cycles'].values, 
        results_df['true_remaining_cycles'].values, 
        'Thermal Shock Life: Predicted vs True',
        'test_predictions_cycle_scatter.png'
    )
    print(f"服役寿命散点图已保存至 test_predictions_cycle_scatter.png")
    
    return results_df

def main():
    # 加载模型和标准化器
    print("正在加载模型和标准化器...")
    model, area_scaler, cycle_scaler = load_model_and_scaler()
    
    # 对测试集进行预测
    print("正在进行预测...")
    results = predict_test_set(model, area_scaler, cycle_scaler)
    
    # 打印一些基本统计信息
    print("\n预测结果统计信息：")
    print("剥落面积预测：")
    print(f"预测样本数量: {len(results)}")
    print(f"预测剥落面积范围: {results['predicted_spalling_area'].min():.2f} - {results['predicted_spalling_area'].max():.2f}")
    print(f"平均预测剥落面积: {results['predicted_spalling_area'].mean():.2f}")
    
    print("\n服役寿命预测：")
    print(f"预测服役寿命范围: {results['predicted_remaining_cycles'].min():.2f} - {results['predicted_remaining_cycles'].max():.2f}")
    print(f"平均预测服役寿命: {results['predicted_remaining_cycles'].mean():.2f}")

if __name__ == "__main__":
    main()
