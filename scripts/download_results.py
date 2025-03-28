import os
import gdown


if __name__ == "__main__":
    id = "1BI2bH0zHuQ5yOf4gDy_S-58I8B0Inifm"
    url = f"https://drive.google.com/uc?id={id}"
    output = "results.zip"
    gdown.download(url, output, quiet=False)
    os.system("unzip results.zip")
    os.system("rm results.zip")
    print("Downloaded and unzipped results")
