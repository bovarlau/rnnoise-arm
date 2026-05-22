# 移植RNNoise项目
    RNNoise是效果很好的降噪算法库,其采用了DSP+RNN的处理方式,本项目在源码的基础上,对嵌入式或者低端ARM平台做了极致移植优化,使得在ARM平台能满足实时降噪要求.已验证平台全志A133,瑞芯微RK3568. 
    RTF做到0.6左右.

## 文件目录
    论文文档: DNN-Denoise.1709.08243v3.pdf
    论文阐述: https://jmvalin.ca/demo/rnnoise/       
    原版仓库: https://github.com/xiph/rnnoise 
      
    src        -- 源码库,针对ARM平台做了算法优化,满足ARM平台的实时降噪要求.                   
    include    -- 头文件
    examples   -- 测试例程
    Android.bp -- Android平台的编译脚本,编译后在Android的HAL层音频库里添加该算法库的使用.

## 模型文件
    模型文件为src/rnnoise_data.c,有其它平台的移植优化需求请联系19471477@qq.com
