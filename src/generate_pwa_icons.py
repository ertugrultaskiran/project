"""
Generate PWA Icons
==================
Creates app icons in multiple sizes for PWA
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path):
    """Create a simple app icon"""
    # Create base image with gradient background
    img = Image.new('RGB', (size, size), color='#4F46E5')
    draw = ImageDraw.Draw(img)
    
    # Add gradient effect
    for i in range(size):
        alpha = i / size
        color = (
            int(79 + (139 - 79) * alpha),   # R
            int(70 + (92 - 70) * alpha),     # G
            int(229 + (246 - 229) * alpha)   # B
        )
        draw.rectangle([0, i, size, i+1], fill=color)
    
    # Add icon text/symbol
    try:
        # Try to use a nice font
        font_size = size // 3
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    except:
        font = None
    
    # Draw ticket symbol/text
    text = "🎫"
    if font:
        # Get text size
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center text
        x = (size - text_width) // 2
        y = (size - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
    else:
        # Draw simple geometric shape if font fails
        margin = size // 4
        draw.rounded_rectangle(
            [margin, margin, size-margin, size-margin],
            radius=size//10,
            outline='white',
            width=size//20
        )
        
        # Add "IT" text in center
        center_x = size // 2
        center_y = size // 2
        text_size = size // 4
        
        # Draw "IT" manually with simple rectangles
        # Letter "I"
        i_width = text_size // 4
        i_x = center_x - text_size // 2
        draw.rectangle([i_x, center_y - text_size//2, i_x + i_width, center_y + text_size//2], fill='white')
        
        # Letter "T"
        t_x = center_x + text_size // 4
        draw.rectangle([t_x - text_size//3, center_y - text_size//2, t_x + text_size//3, center_y - text_size//2 + i_width], fill='white')
        draw.rectangle([t_x - i_width//2, center_y - text_size//2, t_x + i_width//2, center_y + text_size//2], fill='white')
    
    # Save icon
    img.save(output_path, 'PNG', quality=95)
    print(f"[OK] Created icon: {output_path} ({size}x{size})")

def main():
    """Generate all required icon sizes"""
    # Create icons directory
    icons_dir = os.path.join('static', 'icons')
    os.makedirs(icons_dir, exist_ok=True)
    
    # Icon sizes required for PWA
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    
    print("Generating PWA Icons...")
    print("=" * 60)
    
    for size in sizes:
        output_path = os.path.join(icons_dir, f'icon-{size}x{size}.png')
        create_icon(size, output_path)
    
    print("=" * 60)
    print(f"[OK] Generated {len(sizes)} icons successfully!")
    print(f"Location: {icons_dir}")
    
    # Create favicon
    favicon_path = os.path.join('static', 'favicon.ico')
    create_icon(32, 'temp_favicon.png')
    
    # Convert to ICO
    try:
        img = Image.open('temp_favicon.png')
        img.save(favicon_path, format='ICO', sizes=[(32, 32)])
        os.remove('temp_favicon.png')
        print(f"[OK] Created favicon: {favicon_path}")
    except Exception as e:
        print(f"[WARNING] Favicon creation failed: {e}")
    
    print("\n[SUCCESS] Icon generation complete!")

if __name__ == '__main__':
    # Change to src directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

