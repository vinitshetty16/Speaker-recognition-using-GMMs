# Speaker Recognition using GMMs
<img src="https://github.com/vinitshetty16/Speaker-recognition-using-GMMs/assets/63487624/2e9b6bc9-6090-4dc3-9c74-fb9ced37ea48" alt="image" width="1400px">

## Methodology

The methodology for speaker recognition using Gaussian Mixture Models (GMMs) involves the following steps:

- **Building GMMs with MFCC features**: This step involves extracting Mel-Frequency Cepstral Coefficients (MFCC) features from audio files, building GMMs using these features, and saving the GMM models to files.

- **Speaker Recognition using GMMs**: In this step, the pre-trained GMM models are loaded, and MFCC features are extracted from test audio files. These features are then used to identify the speaker by calculating scores from each GMM model and selecting the speaker with the highest score.

## Dependencies

- NumPy
- Librosa
- Pydub
- Scikit-learn

## Conclusion

The speaker recognition algorithm based on GMMs achieved an overall recognition accuracy of 92.57% when evaluated on the complete test dataset. This indicates the effectiveness of the model in accurately predicting speakers from audio samples. By leveraging MFCC features and GMMs, the algorithm demonstrates the application of machine learning techniques in speaker recognition tasks. 

## Observations

- The algorithm accurately predicted the majority of speakers with a high level of accuracy.
- The use of GMMs with MFCC features proved to be effective in capturing speaker characteristics and distinguishing between different speakers.
- Further optimizations and refinements could potentially improve the recognition accuracy even further.
