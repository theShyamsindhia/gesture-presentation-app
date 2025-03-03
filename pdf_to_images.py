import os
import tkinter as tk
from tkinter import filedialog
from pdf2image import convert_from_path
import cv2
import numpy as np

def convert_pdf_to_images():
    # Create output directory if it doesn't exist
    output_dir = "presentation_slides"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Setup tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Ask user to select PDF file
    pdf_path = filedialog.askopenfilename(
        title="Select PDF Presentation",
        filetypes=[("PDF files", "*.pdf")]
    )
    
    if not pdf_path:
        print("No PDF selected. Exiting...")
        return
    
    try:
        print(f"Converting PDF: {pdf_path}")
        print("This may take a moment...")
        
        # Convert PDF to images
        pages = convert_from_path(pdf_path)
        
        # Save each page as an image
        for i, page in enumerate(pages):
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            
            # Save the image
            output_path = os.path.join(output_dir, f"slide_{i+1:03d}.jpg")
            cv2.imwrite(output_path, opencv_image)
            print(f"Saved slide {i+1} to {output_path}")
        
        print(f"\nConversion complete!")
        print(f"Images saved in: {os.path.abspath(output_dir)}")
        print(f"Total slides converted: {len(pages)}")
        print("\nYou can now run presentation_app.py and select this folder!")
        
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have poppler installed:")
        print("   - Windows: Download from http://blog.alivate.com.au/poppler-windows/")
        print("   - Extract the downloaded file")
        print("   - Add the bin folder to your system PATH")
        print("2. Try closing any programs that might be using the PDF")
        print("3. Make sure you have write permissions in the output directory")

if __name__ == "__main__":
    convert_pdf_to_images()
