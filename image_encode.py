import base64
from mimetypes import guess_type

# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def local_file_to_data_url(file_path, mime_type=None):
    """Encode a local file (e.g. PDF) into a data URL for Mistral Document AI.
    Only base64 is supported by Document AI; URLs are not supported.
    """
    if mime_type is None:
        mime_type, _ = guess_type(file_path)
    if mime_type is None:
        if file_path.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        else:
            mime_type = "application/octet-stream"
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{data}"

