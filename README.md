# Sleep Stage Classification and Deep Analysis using PPG Signal

This project focuses on the classification and analysis of sleep stages using Photoplethysmography (PPG) signals. The research aims to develop accurate methods for sleep disorder detection and classification through advanced signal processing and machine learning techniques.

## Project Overview

Sleep disorders affect millions of people worldwide and can lead to serious health problems if left undiagnosed and untreated. Traditional sleep monitoring methods like polysomnography (PSG) require specialized equipment and clinical settings. This project explores the use of PPG signals, which can be easily acquired using wearable devices, as an alternative approach for sleep stage classification and sleep disorder detection.

## Workflow

The project follows a comprehensive workflow for sleep stage classification:

![Project Workflow](Workflow.png)

1. **Problem Definition**: Identifying the need for accessible sleep stage classification methods
2. **Data Collection**: Gathering PPG signal data from subjects during sleep
3. **Data Preprocessing**: Cleaning and preparing the PPG signals for analysis
4. **Exploratory Data Analysis (EDA)**: Understanding patterns and characteristics in the data
5. **Model Selection**: Choosing appropriate machine learning and deep learning models
6. **Model Training**: Training the selected models on the preprocessed data
7. **Model Evaluation**: Assessing model performance using appropriate metrics
8. **Model Optimization**: Fine-tuning models to improve accuracy
9. **Model Deployment**: Implementing the models for practical use
10. **Monitoring and Maintenance**: Ensuring continued model performance
11. **Documentation and Reporting**: Recording methodologies and findings
12. **Feedback Loop**: Incorporating feedback for continuous improvement

## Methodologies

The project employs several advanced signal processing and machine learning techniques:

### Signal Processing Techniques
- **Discrete Wavelet Transform (DWT)**: For time-frequency analysis of PPG signals
- **Empirical Mode Decomposition (EMD)**: For adaptive signal decomposition
- **Heart Rate Variability (HRV) Analysis**: For extracting cardiac features

### Machine Learning Models
- **VGGNet-based Deep Learning**: Custom neural network architecture for PPG signal classification
- **Multi-class Classification**: For distinguishing between different sleep stages and disorders

## Project Structure

- **hrv/**: Jupyter notebooks for heart rate variability analysis and model implementation
  - `PPGVGGNet.ipynb`: Implementation of VGG-based neural network for PPG classification
  - `data_analysis.ipynb`: Exploratory data analysis of PPG signals
  - `dwt_ppg.ipynb`: Discrete Wavelet Transform analysis of PPG signals
  - `emd_ppg.ipynb`: Empirical Mode Decomposition analysis of PPG signals

- **matlab codes/**: MATLAB implementations for signal processing and data extraction
  - Data extraction from EDF files
  - Signal visualization
  - Wavelet scattering analysis
  - PPG signal preprocessing

## Sleep Stage Classification

The project focuses on classifying sleep into different stages:
- Wake
- Light Sleep (N1, N2)
- Deep Sleep (N3)
- REM Sleep

Additionally, the system aims to detect sleep disorders such as:
- Sleep Apnea
- Insomnia
- Periodic Limb Movement Disorder
- Restless Leg Syndrome

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- Matplotlib
- EMD-signal
- MATLAB (for some preprocessing steps)

## Future Work

- Integration with wearable devices for real-time sleep monitoring
- Expansion of the dataset to improve model generalization
- Development of a user-friendly interface for sleep analysis
- Validation in clinical settings

## License

This project is licensed under the terms of the included LICENSE file.
