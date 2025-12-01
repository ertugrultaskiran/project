"""
Simple Flask Test
Just to verify Flask works
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return """
    <html>
    <head><title>Test</title></head>
    <body style="font-family: Arial; padding: 50px; text-align: center;">
        <h1 style="color: green;">✅ FLASK ÇALIŞIYOR!</h1>
        <p>Eğer bunu görüyorsanız, Flask düzgün çalışıyor demektir.</p>
        <p>Şimdi ana uygulamayı test edebilirsiniz.</p>
    </body>
    </html>
    """

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧪 FLASK TEST UYGULAMASI")
    print("="*50)
    print("\n📍 Tarayıcıda aç:")
    print("   http://127.0.0.1:5001")
    print("\n⏹  CTRL+C ile durdur")
    print("="*50 + "\n")
    
    app.run(host='127.0.0.1', port=5001, debug=False)

