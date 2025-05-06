import cloudinary
from cloudinary.uploader import upload

# Configure Cloudinary (replace with your credentials)
cloudinary.config(
    cloud_name='dqcgb73mf',
    api_key='711966464192934',
    api_secret='_6CYk7HyN4lF9ZG2NDjWrwAvDAw'
)
def upload_json_to_cloudinary(file_path, upload_preset="Matches"):
    try:
        # Upload JSON file using the Matches preset
        response = upload(
            file_path,
            resource_type="raw",              # Use 'raw' for JSON files
            upload_preset=upload_preset       # Specify the Matches upload preset
        )
        # Return the secure URL
        return response['secure_url']
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

# Example usage
file_path = "player_analysis.json"  # Replace with actual file path
url = upload_json_to_cloudinary(file_path)
