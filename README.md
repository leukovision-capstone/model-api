# Flask API for Image Analysis and Detection

This API is built using Flask and provides two endpoints for image analysis and detection using Keras and YOLOv8 models.

## Features

- **/analyze**: Analyzes an image and returns the predicted class along with the confidence level.
- **/detect**: Detects objects in an image and returns the annotated image.

## Prerequisites

Before running the API, ensure you have the following:

- Python 3.7 or newer
- pip (Python package manager)

## Installation

1. **Clone this repository**:
```bash
git clone https://github.com/leukovision-capstone/model-api.git cd model-api
```
2. **Create and activate a virtual environment** (optional but recommended):
```bash
python3 -m venv venv
```
**Activate the virtual environment**
```bash
source venv/bin/activate
```
3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
4. **Ensure the required models are available**:
   - Keras Model: `model_leukovision.keras`
   - YOLO Model: `leukovision.pt`
   - Place these models in the `./model/` folder.
## Running the API
Once all dependencies are installed and the models are available, you can run the API with the following command:
```bash
python app.py
```
The API will run at `http://0.0.0.0:5001`.

## Testing the API

You can test the API using tools like Postman or `curl`.

### Endpoint `/analyze`

**Request**:
- Method: `POST`
- URL: `http://0.0.0.0:5001/analyze`
- Body: Form-data with key `image` and the image file as the value.

**Example using curl**:
```bash
curl -X POST -F "image=@path_to_your_image.jpg" http://0.0.0.0:5001/analyze
```
**Response**
```json
{
  "status": "success",
  "message": "Model is analyze successfully",
  "data": {
    "confidence": 0.7316434383392334,
    "predicted_class": "Pro_Malignant_Pro-B"
  }
}
```
### Endpoint `/detect`

**Request**:
- Method: `POST`
- URL: `http://127.0.0.1:5001/detect`
- Body: Form-data with key `image` and the image file as the value.

**Example using curl**:
```bash
curl -X POST -F "image=@path_to_your_image.jpg" http://0.0.0.0:5001/detect
```
**Response**:
An annotated image will be returned as the response.

## Notes

- Make sure to test the API after installing new packages or updating models.
- If you encounter issues, check the logs for more information.

## License

MIT License
