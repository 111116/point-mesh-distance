import requests

def download_bunny_model(url, filename):
    """
    Download the bunny model and save it as an .obj file.

    Parameters:
        url (str): The URL of the bunny model.
        filename (str): The name of the file to save the model to.

    Returns:
        None
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception if the GET request failed.

    with open(filename, 'wb') as f:
        f.write(response.content)

bunny_url = "https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj"
output_filename = "bunny.obj"

download_bunny_model(bunny_url, output_filename)