from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import socket

app = FastAPI(title="Pest Detection API")

# CORS middleware for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pest classes mapping
CLASSES = {
    0: "Dolycoris baccarum",
    1: "Lycorma delicatula",
    2: "Eurydema dominulus",
    3: "Pieris rapae",
    4: "Halyomorpha halys",
    5: "Spilosoma obliqua",
    6: "Graphosoma rubrolineata",
    7: "Luperomorpha suturalis",
    8: "Leptocorisa acuta",
    9: "Sesamia inferens",
    10: "Cicadella viridis",
    11: "Callitettix versicolor",
    12: "Scotinophara lurida",
    13: "Cletus punctiger",
    14: "Nezara viridula",
    15: "Dicladispa armigera",
    16: "Riptortus pedestris",
    17: "Maruca testulalis",
    18: "Chauliops fallax",
    19: "Chilo supperssalis",
    20: "Stollia ventralis",
    21: "Nilaparvata lugens",
    22: "Diostrombus politus",
    23: "Phyllotreta striolata",
    24: "Aulacophora indica",
    25: "Laodelphax striatellus",
    26: "Ceroplastes ceriferus",
    27: "Corythucha marmorata",
    28: "Dryocosmus Kuriphilus",
    29: "Porthesia taiwana",
    30: "Chromatomyia horticola",
    31: "Iscadia inexacta",
    32: "Plutella xylostella",
    33: "Empoasca flavescens",
    34: "Dolerus tritici",
    35: "Spodoptera litura",
    36: "Corythucha ciliata",
    37: "Bemisia tabaci",
    38: "Ceutorhynchus asper",
    39: "Strongyllodes variegatus"
}

# Pesticides mapping for each pest
PESTICIDES = {
    "Dolycoris baccarum": [
        "Cyfluthrin 5% EW",
        "Lambda-Cyhalothrin",
        "Bifenthrin"
    ],
    "Lycorma delicatula": [
        "Imidacloprid",
        "Dinotefuran",
        "Bifenthrin",
        "Carbaryl"
    ],
    "Eurydema dominulus": [
        "Cyfluthrin",
        "Lambda-Cyhalothrin",
        "Bifenthrin",
        "Neem Oil",
        "Insecticidal Soap",
        "Pyrethrin-Based Spray"
    ],
    "Pieris rapae": [
        "Spinosad",
        "Pyrethroids (Permethrin)",
        "Carbamates",
        "Neem Oil",
        "Insecticidal Soap",
        "Bacillus thuringiensis (Bt)"
    ],
    "Halyomorpha halys": [
        "Bifenthrin",
        "Cyfluthrin",
        "Carbamates",
        "Organophosphates",
        "Insecticidal Soaps",
        "Neem Oil",
        "Pyrethrin-Based Sprays",
        "Kaolin Clay"
    ],
    "Spilosoma obliqua": [
        "Spinosad",
        "Bifenthrin",
        "Bacillus thuringiensis (Bt)",
        "Neem Oil",
        "Pyrethrin-Based Sprays"
    ],
    "Graphosoma rubrolineata": [
        "Bifenthrin",
        "Carbamates",
        "Organophosphates",
        "Neem Oil",
        "Pyrethrin-Based Sprays"
    ],
    "Luperomorpha suturalis": [
        "Bifenthrin",
        "Organophosphates",
        "Neem Oil",
        "Pyrethrin-Based Sprays"
    ],
    "Leptocorisa acuta": [
        "Deltamethrin",
        "Cypermethrin",
        "Neonicotinoids"
    ],
    "Sesamia inferens": [
        "Cypermethrin",
        "Deltamethrin",
        "Organophosphates",
        "Neonicotinoids"
    ],
    "Cicadella viridis": [
        "Bifenthrin",
        "Carbamates",
        "Neonicotinoids",
        "Neem Oil",
        "Insecticidal Soaps",
        "Pyrethrin-Based Sprays"
    ],
    "Callitettix versicolor": [
        "Bifenthrin",
        "Carbamates",
        "Neonicotinoids",
        "Neem Oil",
        "Pyrethrin-Based Sprays"
    ],
    "Scotinophara lurida": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Bacillus thuringiensis (Bt)"
    ],
    "Cletus punctiger": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Nezara viridula": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Dicladispa armigera": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Riptortus pedestris": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Maruca testulalis": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Chauliops fallax": [
        "Chlorpyriphos (0.02%)",
        "Monocrotophos (0.04%)",
        "Fenitrothion (0.05%)",
        "Cypermethrin (0.005%)",
        "Fenvalerate (0.05%)",
        "Dimethoate (0.03%)"
    ],
    "Chilo supperssalis": [
        "Cypermethrin",
        "Deltamethrin",
        "Lambdacyhalothrin (4.9% CS)",
        "Chlorpyriphos (50%) + Cypermethrin (5%)",
        "Cartap Hydrochloride (4% GR)"
    ],
    "Stollia ventralis": [
        "Cypermethrin",
        "Deltamethrin",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Nilaparvata lugens": [
        "Imidacloprid",
        "Chlorpyrifos",
        "Fipronil",
        "Bifenthrin"
    ],
    "Diostrombus politus": [
        "Organophosphates",
        "Pyrethroids",
        "Neonicotinoids"
    ],
    "Phyllotreta striolata": [
        "Carbaryl",
        "Imidacloprid",
        "Metarhizium anisopliae"
    ],
    "Aulacophora indica": [
        "Carbaryl",
        "Imidacloprid",
        "Metarhizium anisopliae"
    ],
    "Laodelphax striatellus": [
        "Triflumezopyrim",
        "Carbaryl",
        "Imidacloprid",
        "Metarhizium anisopliae"
    ],
    "Ceroplastes ceriferus": [
        "Chlorpyrifos",
        "Malathion",
        "Imidacloprid",
        "Neonicotinoids"
    ],
    "Corythucha marmorata": [
        "Bifenthrin",
        "Pyrethroids",
        "Organophosphates",
        "Neem Oil"
    ],
    "Dryocosmus Kuriphilus": [
        "Imidacloprid",
        "Acetamiprid",
        "Neonicotinoids"
    ],
    "Porthesia taiwana": [
        "Spinosad",
        "Bacillus thuringiensis (Bt)",
        "Neem Oil"
    ],
    "Chromatomyia horticola": [
        "Abamectin",
        "Cyromazine",
        "Imidacloprid"
    ],
    "Iscadia inexacta": [
        "Spinosad",
        "Bacillus thuringiensis (Bt)",
        "Neem Oil"
    ],
    "Plutella xylostella": [
        "Spinosad",
        "Abamectin",
        "Emamectin Benzoate",
        "Bacillus thuringiensis (Bt)"
    ],
    "Empoasca flavescens": [
        "Imidacloprid",
        "Bifenthrin",
        "Fipronil"
    ],
    "Dolerus tritici": [
        "Lambda-Cyhalothrin",
        "Cypermethrin",
        "Deltamethrin"
    ],
    "Spodoptera litura": [
        "Spinosad",
        "Emamectin Benzoate",
        "Chlorantraniliprole",
        "Neem Oil"
    ],
    "Corythucha ciliata": [
        "Bifenthrin",
        "Pyrethroids",
        "Organophosphates"
    ],
    "Bemisia tabaci": [
        "Imidacloprid",
        "Thiamethoxam",
        "Spiromesifen"
    ],
    "Ceutorhynchus asper": [
        "Pyrethroids",
        "Carbamates",
        "Neonicotinoids"
    ],
    "Strongyllodes variegatus": [
        "Cypermethrin",
        "Deltamethrin",
        "Neonicotinoids"
    ]
}

# Model definition
class InsectModel(nn.Module):
    def __init__(self, num_classes):
        super(InsectModel, self).__init__()
        self.num_classes = num_classes
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, image):
        return self.model(image)

# Image transformation
def get_transform():
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

# Helper function to get local IP
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# Global variables for model
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = get_transform()

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    try:
        model = InsectModel(num_classes=40)
        model.load_state_dict(torch.load("pest_model.pth", map_location=device))
        model.to(device)
        model.eval()
        local_ip = get_local_ip()
        print(f"✅ Model loaded successfully on {device}")
        print(f"🌐 Server running on: http://{local_ip}:8000")
        print(f"📱 Use this in your Android app: http://{local_ip}:8000/predict")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    local_ip = get_local_ip()
    return {
        "status": "online",
        "message": "Pest Detection API is running",
        "device": str(device),
        "server_ip": local_ip,
        "endpoints": {
            "predict": f"http://{local_ip}:8000/predict",
            "classes": f"http://{local_ip}:8000/classes"
        }
    }

@app.post("/predict")
async def predict_pest(file: UploadFile = File(...)):
    """
    Predict pest from uploaded image
    
    Args:
        file: Image file (jpg, jpeg, png)
    
    Returns:
        JSON with predicted class, confidence, top 3 predictions, and pesticides
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        # Preprocess
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_np /= 255.0
        
        # Apply transforms
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3, dim=1)
            
            predicted_class = top3_idx[0][0].item()
            confidence = top3_prob[0][0].item()
            predicted_name = CLASSES[predicted_class]
            
            # Check if confidence is too low (threshold: 50%)
            CONFIDENCE_THRESHOLD = 0.50
            is_pest = confidence >= CONFIDENCE_THRESHOLD
            
            if not is_pest:
                # Not a pest detected
                response = {
                    "success": True,
                    "is_pest": False,
                    "predicted_class": -1,
                    "predicted_name": "Not a Pest",
                    "confidence": float(confidence),
                    "message": "This image does not appear to be a pest. Please upload a clearer image of the pest.",
                    "pesticides": [],
                    "top_3_predictions": [
                        {
                            "class": top3_idx[0][i].item(),
                            "name": CLASSES[top3_idx[0][i].item()],
                            "confidence": float(top3_prob[0][i].item())
                        }
                        for i in range(3)
                    ]
                }
            else:
                # Valid pest detected
                pesticides = PESTICIDES.get(predicted_name, ["No pesticide data available"])
                
                response = {
                    "success": True,
                    "is_pest": True,
                    "predicted_class": predicted_class,
                    "predicted_name": predicted_name,
                    "confidence": float(confidence),
                    "pesticides": pesticides,
                    "top_3_predictions": [
                        {
                            "class": top3_idx[0][i].item(),
                            "name": CLASSES[top3_idx[0][i].item()],
                            "confidence": float(top3_prob[0][i].item())
                        }
                        for i in range(3)
                    ]
                }
            
            print(f"✅ Prediction: {predicted_name} ({confidence*100:.2f}%)")
            if is_pest:
                print(f"💊 Pesticides: {', '.join(pesticides[:3])}")
            else:
                print(f"⚠️ Not a pest detected - Low confidence")
            return JSONResponse(content=response)
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get all available pest classes"""
    return {
        "total_classes": len(CLASSES),
        "classes": CLASSES
    }

@app.get("/pesticides/{pest_name}")
async def get_pesticides(pest_name: str):
    """Get pesticides for a specific pest"""
    pesticides = PESTICIDES.get(pest_name, [])
    if not pesticides:
        raise HTTPException(status_code=404, detail="Pest not found")
    return {
        "pest_name": pest_name,
        "pesticides": pesticides
    }

if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🐛 PEST DETECTION API SERVER")
    print("="*60)
    print(f"📱 Mobile App URL: http://{local_ip}:8000/predict")
    print(f"🌐 Browser Test: http://{local_ip}:8000")
    print(f"💻 Localhost: http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)