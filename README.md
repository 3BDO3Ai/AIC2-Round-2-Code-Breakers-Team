# Speech-to-Text (STT) System for Egyptian Dialect Using Wav2Vec2

## Project Overview

### Summary
This project aims to develop a robust Speech-to-Text (STT) system specifically designed for transcribing Egyptian dialect speech into text. The system leverages the pre-trained model facebookwav2vec2-xls-r-300m and employs phased training with a large dataset to achieve accurate transcription.

### Main Purpose
The primary objective is to enhance speech recognition capabilities for the Egyptian dialect through fine-tuning of the Wav2Vec2 model, ensuring high accuracy and reliability.

## Installation Instructions

### Dependencies
- Python 3.11
- Jupyter Notebook
- Required Libraries `datasets`, `torchaudio`, `transformers`, `pandas`, `numpy`, `librosa`, `torch`, etc.
- Optional Kaggle account for TPU usage

### Setup Steps
1. Download the codes
   

2. Install dependencies using pip
   ```bash
   pip install -r requirements.txt
   ```

3. Download and extract the dataset
   - Download the dataset and extract it into the respective directories (`train` and `adapt`).

## Data and Resources

Data and Resources
Dataset Information
MTC-ASR-Dataset-16K
Arabic Speech Recognition for Egyptian Dialects using a data resource consisting of 100 hrs recorded Egyptian dialect samples.

## Testing dataset
Duration: 3 hours
Number of files: 1726 audio 16KHz WAV files
Environment: Clean and Noisy
## Dataset Structure
train/: Contains the training audio data (WAV files).
adapt/: Contains the adaptation audio data (WAV files).
train.csv: CSV file with wav_id and transcription columns for training data.
adapt.csv: CSV file with wav_id and transcription columns for adaptation data.

### Accessing the Dataset
Download the dataset from the provided link and extract it into the respective directories (`train` and `adapt`).

## Code Structure

### Organization
- Preprocess_&_Prepare_Data.ipynb Handles data preprocessing and feature extraction.
- Train.ipynb Defines and trains the Wav2Vec2 model.
- Test.ipynb Loads the trained model and evaluates it on test data.

### Key Scripts
- Preprocess_&_Prepare_Data.ipynb Preprocesses audio files, extracts features, and saves processed data.
- Train.ipynb Defines model architecture, sets training parameters, trains the model, and saves checkpoints.
- Test.ipynb Loads trained model, transcribes audio files, and evaluates performance metrics.

## Running the Code

### Steps to Run
1. Preprocess data Execute all cells in `Preprocess_&_Prepare_Data.ipynb` on Kaggle TPU.
2. Download preprocessed data Transfer data from Kaggle to your local machine.
3. Train the model Execute all cells in `Train.ipynb` on your local machine, using phased training due to the large dataset.
4. Evaluate model performance Execute all cells in `Test.ipynb` on your local machine.

### Example Commands
- To preprocess data Execute all cells in `Preprocess_&_Prepare_Data.ipynb` on Kaggle.
- To train the model Execute all cells in `Train.ipynb` on your local machine.
- To test the model Execute all cells in `Test.ipynb` on your local machine.

## Model Details

### Model Architecture
- Utilizes pre-trained Wav2Vec2 model facebookwav2vec2-xls-r-300m for speech recognition.
- Fine-tuned on provided dataset for Egyptian dialect through a phased training approach.

### Training Process
- Dataset split and trained in phases to manage large size effectively.
- Further training on entire dataset by sampling approximately 40% randomly to enhance Word Error Rate (WER).
- 
### Final Model
- Model trained and uploaded to hugging face: 3BDOAi3/facebookwav2vec2-xls-r-300m-finetuned-with-MTC-Dataset

### Checkpoints
- Model checkpoints saved during training process to designated output directory.
- Best Model checkpoint is on the next link: https://drive.google.com/drive/folders/1iaDpyfDGHSdddQlzKit0Tq6cujJXweT2?usp=drive_link

## Results and Evaluation

### Performance Evaluation
- Model evaluated using metrics such as Word Error Rate (WER).

### Metrics
- Word Error Rate (WER) used as the primary metric for evaluating transcription accuracy.

## Additional Information

### Assets and Resources
- Ensure correct saving of model checkpoints and processed datasets.
- Include inference scripts to ensure reproducibility of results.

### Tips
- Regularly save model checkpoints to prevent data loss during training.
- Monitor training and validation metrics to avoid overfitting and ensure model robustness.
- **Optimal Environment:** Preprocess the data on Kaggle's TPU for efficiency, and train the model on a high-performance computer,and i used (PC with GPU A4000 16GB and 135 GB RAM) for faster processing.
- **Path Management:** Ensure consistency with file paths across scripts and notebooks to avoid errors during data loading, model training, and evaluation.

## Our Teeam
### Code Breakers

Abdelrahman Ahmed Karim Mohammed

Mohammed Usama Elhagari

Manar Ashraf Ibrahim Eldesouky

Ahmed Mohamed Mahmoud Elsayed

Youssef Hatem Abd-Elmasen

New Mansoura University

LEVEL 2 & LEVEL 3

Faculty of Computer Science and Engineering

Artificial Intelligence Engineering Department

Artificial Intelligence Science Department

Computer Engineering Department
