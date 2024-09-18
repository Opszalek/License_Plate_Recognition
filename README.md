# **License Plate Recognition**

This project implements a license plate recognition system using traditional image processing techniques, without relying on OCR libraries or machine learning models. 
The system works under the following conditions:

- The plate occupies at least one-third of the image width.
- The angle between the camera’s optical axis and the plane of the plate is no more than 45 degrees.
- Recognizes standard plates with black characters on a white background (7 or 8 characters).
- Plates from various regions are supported.
- Images can have varying resolutions.

### Example images:

<p align="center">
  <img src="https://github.com/Opszalek/Image_to_readme/blob/main/License_Plate_Recognition/459011288_1970877216659435_8483961446294724982_n.jpg?raw=true" alt="Example 1" width="300"/>
  <img src="https://github.com/Opszalek/Image_to_readme/blob/main/License_Plate_Recognition/459315184_1071610051246153_2361377541037099992_n.jpg?raw=true" alt="Example 2" width="300"/>
  <img src="https://github.com/Opszalek/Image_to_readme/blob/main/License_Plate_Recognition/459451150_2302604903407199_7358595598509359465_n.jpg?raw=true" alt="Example 3" width="300"/>
</p>

### Results:
```json
{
    "459011288_1970877216659435_8483961446294724982_n.jpg": "PTUJE07",
    "459315184_1071610051246153_2361377541037099992_n.jpg": "PTU8166C",
    "459451150_2302604903407199_7358595598509359465_n.jpg": "PTU4806F"
}
```

### Predict plate numbers:
```bash
python3 predict_plate_main.py /path/to/images name/for/results/file.json
```

### Calculate accuracy:
```bash
python3 checkAccuracy.py /path/to/correct/results.json /path/to/results.json
```


### Example dataset:
[Dataset link](https://drive.google.com/drive/folders/1NBZWRr6RhqnqlxofhMI6buYDBfdTCwOi?usp=sharing)
