# 时序与问卷联合分类及条件遮蔽重建

当前版本采用 MedTVL 第3.4节的共享专家与稀疏路由MoE融合运动时序与问卷，运动分支包含归一化前的幅度特征。
训练和预测疾病均输入受试者的运动时序和完整30题问卷。

## 模型

- MOMENT基座和patch embedding冻结，训练q/v LoRA，rank64、alpha32。
- 分类分支保留132个通道，每通道768→16投影后拼接，经2112→128得到运动特征。
- 完整问卷经30→32编码，作为分类头的问卷输入。
- 启用use_amplitude_features时，在RevIN前按11动作×2手腕×2传感器计算44维幅度。每个值为log(sqrt(三轴时间方差之和)+1e-6)，时间方差使用总体方差。
- 训练集逐维均值和总体标准差用于幅度标准化，标准差下限1e-6。统计量保存在检查点configuration中的amplitude_mean和amplitude_std，预测自动复用。
- fusion_dim=128。128维运动与44维标准化幅度合并后经172→128投影，32维问卷经32→128投影，两者均使用GELU和LayerNorm。use_amplitude_features=False时运动投影为128→128。
- 两侧128维表示拼接为256维联合表示，再按MedTVL第3.4节公式（5）进行MoE融合。一个共享FFN由标量sigmoid门控，moe_num_experts个路由FFN由softmax后top-k选择，仅对选中的样本执行专家。保留完整softmax中选中专家的原始概率。
- 默认moe_num_experts=4、moe_top_k=2、moe_expert_hidden_dim=256，专家输入256维、隐藏层256维、输出128维，经128→64→3分类器。共享和路由专家均使用SiLU门控FFN，计算为down_proj(SiLU(gate_proj(z))⊙up_proj(z))。专家的三个线性层、路由层和共享专家门控层均为bias=False。
- 辅助分支每批独立为每人随机遮蔽9/30题。
- MOMENT先沿时间片段求均值，再对132个通道逐维取平均，得到每人一个768维运动向量。
- 可见答案和30维可见标记经60→768得到问卷上下文，与归一化后的运动向量相加。
- 共享上下文与每题768维固定条件拼接，经1536→512→768解码器恢复答案极性向量。
- 隐藏答案在编码前置零。MSE对向量维求和，只在隐藏题上求均值。
- 分类问卷编码器与重建上下文编码器独立，MSE通过辅助运动表示更新LoRA。

## 运行

在train.py的CONFIG配置路径和超参数，然后运行：

    python train.py

可通过命令行指定输出目录：

    python train.py --run-dir outputs/questionnaire_amplitude_moe_masked_mse

默认20轮、seed42、micro batch8、effective batch32。
每轮训练和验证结束后统一打印指标。
损失为训练集类别加权交叉熵 + reconstruction_weight × masked_MSE，默认系数0.01。
LoRA学习率5e-5，分类头和辅助解码器学习率1e-3，按全部训练预算余弦衰减。

默认输出outputs/questionnaire_amplitude_moe_masked_mse。
每轮按验证macro-F1保存最优可训练权重，训练结束恢复最优轮后评估测试。
已有运行目录拒绝覆盖。检查点用于推理，训练重启从预训练基座开始。

问卷缓存读取outputs/reconstruction_text/reconstruction_features.npz，按subject_id对齐。
缓存缺失或reencode_text=True时调用reconstruction.prepare_text，用本地BioLORD生成缓存。

## 重建评估

验证和测试各自使用固定隐藏题，掩码与受试者ID写入对应JSON文件。
masked_mse_paired记录正确配对时序及可见问卷的隐藏题重建MSE。分类指标使用完整问卷及正确配对时序。

## 预测

将30个0/1答案按训练题序写入JSON列表：

    python predict.py --checkpoint outputs/questionnaire_amplitude_moe_masked_mse/model_trainable.pt --movement dataset/preprocessed/movement/001_ml.bin --answers answers.json

可用--moment-dir指定原始MOMENT基座目录，--device选择cpu或cuda。
检查点configuration记录fusion_type="moe"和fusion_dim，训练及预测均构建MoE融合头。

## 清理归档

修改前的训练、预测、模型与说明文件，以及历史实验脚本、根目录旧metadata.json和code.zip，
归档于D:/paper3_before_cleanup_20261108_140016.zip。
移出的旧文件和Python缓存同时存放于D:/paper3_retired_20261108/。
历史实验结果位于outputs/，论文位于read_paper/，数据和模型权重位于原目录。
