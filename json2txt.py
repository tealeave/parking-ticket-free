import requests

def fetch_and_save_data(url, output_file):
    """Fetch JSON data from a URL and save specific fields to a file."""
    
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    data = response.json()
    
    with open(output_file, 'w') as file:
        file.write('long,lat\n')
        for item in data:
            file.write(f"{item[0]},{item[1]}\n")

if __name__ == "__main__":
    URL = 'https://raw.githubusercontent.com/tealeave/parking-ticket-free/master/test_1.json'
    OUTPUT_FILE = 'test_1.txt'
    
    fetch_and_save_data(URL, OUTPUT_FILE)
