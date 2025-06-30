from metal_font_library import MetalFont

def main():
    font = MetalFont()
    
    # Create one sample image
    sample_text = "AHS06718"
    sample_image = font.generate_text(sample_text, width=350, height=80, font_size=45)
    sample_image.save("metal_font_sample.png")

if __name__ == "__main__":
    main()
