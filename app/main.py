from fastapi import FastAPI, Request, Form, Depends, status, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models import User, Outfit, ChatMessage, ModelImage
from app.database import get_db
from app.chatbot_service import chatbot
from app.color_detection import detect_image_colors, get_primary_color_name
from passlib.context import CryptContext
from fastapi.templating import Jinja2Templates
from app.template_filters import sanitize_image_url
from app.url_middleware import URLSanitizationMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import os
import re
import logging
import time
import requests
import random
import hashlib
import base64
from datetime import date
from app import models
import google.generativeai as genai, json
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper function for color hex mapping
def get_color_hex(color_name):
    """Get hex color for display purposes"""
    color_map = {
        'black': '#000000',
        'white': '#ffffff', 
        'red': '#dc3545',
        'blue': '#007bff',
        'green': '#28a745',
        'yellow': '#ffc107',
        'orange': '#fd7e14',
        'purple': '#6f42c1',
        'pink': '#e83e8c',
        'brown': '#8b4513',
        'gray': '#6c757d',
        'grey': '#6c757d',
        'navy': '#001f3f',
        'maroon': '#800000',
        'olive': '#808000',
        'turquoise': '#40e0d0',
        'beige': '#f5f5dc',
        'silver': '#c0c0c0',
        'gold': '#ffd700'
    }
    return color_map.get(color_name.lower() if color_name else '', '#666666')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTFIT_FOLDER = os.path.join(BASE_DIR, "static", "outfits")
MODEL_IMAGES_FOLDER = os.path.join(BASE_DIR, "static", "model_images")
TRYON_RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "tryon_results")
os.makedirs(OUTFIT_FOLDER, exist_ok=True)
os.makedirs(MODEL_IMAGES_FOLDER, exist_ok=True)
os.makedirs(TRYON_RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}

app = FastAPI()

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory="app/templates")
# Add custom URL sanitization filter
templates.env.filters['sanitize_url'] = sanitize_image_url
# Add helper function to template globals
templates.env.globals["get_color_hex"] = get_color_hex
app.add_middleware(SessionMiddleware, secret_key="supersecretkey")
app.add_middleware(URLSanitizationMiddleware)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- CSS HASHING ----------------
def get_file_hash(filepath: str) -> str:
    """Return short md5 hash of a static file for cache busting."""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:
        return "dev"

css_file_path = os.path.join(BASE_DIR, "static", "style.css")
style_hash = get_file_hash(css_file_path)
templates.env.globals["style_hash"] = style_hash

# ---------------- KIE AI SETTINGS ----------------
API_KEY = os.getenv("KIE_API_KEY")
if not API_KEY:
    raise ValueError("KIE_API_KEY environment variable is required but not set.")
BASE_URL = "https://api.kie.ai/api/v1/gpt4o-image"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ---------------- Prompt Components ----------------
colors = [
    "black", "white", "red", "blue", "green", "yellow", "beige", "navy",
    "maroon", "olive", "purple", "turquoise", "orange", "grey", "brown", "pink"
]

styles = [
    "casual", "formal", "streetwear", "vintage", "bohemian", "athleisure",
    "minimalist", "oversized", "tailored", "trendy", "sporty", "elegant"
]

clothing_items = [
    "t-shirt", "shirt", "hoodie", "sweater", "jacket", "blazer", "coat",
    "jeans", "trousers", "cargo pants", "shorts", "skirt", "dress", "kurta"
]

adjectives = [
    "modern", "trendy", "elegant", "stylish", "classic", "sophisticated",
    "luxury", "comfortable", "minimal", "urban-inspired", "3D rendered"
]

# ---------------- Helper Functions ----------------
def get_current_user_email(request: Request):
    return request.session.get("user_email")

def allowed_file(filename):
    if not filename or '.' not in filename:
        return False
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def secure_filename(filename):
    filename = os.path.basename(filename)
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return filename

def get_user_wardrobe(user, db, category: str = None):
    """Fetch wardrobe items for a user, optionally filter by category"""
    query = db.query(Outfit).filter(Outfit.user_id == user.id)
    if category:
        query = query.filter(Outfit.category == category)
    outfits = query.all()

    wardrobe = {}
    for outfit in outfits:
        wardrobe.setdefault(outfit.category, []).append(outfit)
    return wardrobe

def generate_hybrid_prompt():
    """Generate a rich hybrid prompt for realistic outfit rendering."""
    top = random.choice(clothing_items)
    bottom = random.choice(clothing_items)
    while bottom == top:
        bottom = random.choice(clothing_items)

    prompt = (
        f"A {random.choice(adjectives)} outfit featuring a "
        f"{random.choice(colors)} {top} paired with "
        f"{random.choice(colors)} {bottom} in {random.choice(styles)} style. "
        "Highly realistic, detailed fabric textures, photorealistic fashion photography, "
        "3D rendered, professional studio lighting, isolated on a transparent background, "
        "without mannequin or human model."
    )
    return prompt

def generate_outfit_of_the_day(force_new=False):
    """
    Generate one outfit per day via KIE AI and save it as static/outfits/YYYY-MM-DD.png.
    Returns the URL path to the image (or None on failure).
    
    Args:
        force_new (bool): If True, generate a new outfit even if one exists for today
    """
    today = date.today().isoformat()
    output_file = os.path.join(OUTFIT_FOLDER, f"{today}.png")
    
    # If not forcing new and already generated today, reuse it
    if not force_new and os.path.exists(output_file):
        logger.info(f"Outfit for {today} already exists: {output_file}")
        return f"/static/outfits/{today}.png"
    
    # If forcing new, delete the existing file first
    if force_new and os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.info(f"Removed existing outfit file: {output_file}")
        except Exception as e:
            logger.warning(f"Could not remove existing outfit file: {e}")

    # Hybrid prompt
    prompt = generate_hybrid_prompt()
    payload = {
        "prompt": prompt,
        "size": "1:1",
        "nVariants": 1,
        "isEnhance": True,
        "enableFallback": True
    }

    # Step 1: Kick off generation
    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload, headers=HEADERS, timeout=30)
    except Exception as e:
        logger.error(f"KIE generate request error: {e}")
        return None

    if response.status_code != 200:
        logger.error(f"Failed to start generation: {response.status_code} {response.text}")
        return None

    try:
        task_id = response.json()["data"]["taskId"]
    except Exception as e:
        logger.error(f"Unexpected KIE response: {e} - Body: {response.text}")
        return None

    # Step 2: Poll for completion (max ~2.5 minutes)
    for _ in range(30):
        time.sleep(5)
        try:
            poll = requests.get(f"{BASE_URL}/record-info", params={"taskId": task_id}, headers=HEADERS, timeout=30)
        except Exception as e:
            logger.error(f"KIE poll request error: {e}")
            return None

        if poll.status_code != 200:
            logger.error(f"Polling failed: {poll.status_code} {poll.text}")
            return None

        try:
            status_data = poll.json()["data"]
        except Exception as e:
            logger.error(f"Unexpected KIE poll response: {e} - Body: {poll.text}")
            return None

        status = status_data.get("status")
        if status == "SUCCESS":
            try:
                image_url = status_data["response"]["resultUrls"][0]
            except Exception as e:
                logger.error(f"No resultUrls in success payload: {e} - Data: {status_data}")
                return None

            # Download image
            try:
                img = requests.get(image_url, timeout=60)
                img.raise_for_status()
                with open(output_file, "wb") as f:
                    f.write(img.content)
                logger.info(f"Generated outfit saved: {output_file}")
                return f"/static/outfits/{today}.png"
            except Exception as e:
                logger.error(f"Failed to download/save image: {e}")
                return None

    logger.error("Outfit generation timed out.")
    return None

# ---------------- Routes ----------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

# ---------- SIGNUP ----------
@app.get("/signup", response_class=HTMLResponse)
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request, "error": None})

@app.post("/signup")
async def signup(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    height: int = Form(...),
    db: Session = Depends(get_db),
):
    existing = db.query(User).filter(User.email == email.lower()).first()
    if existing:
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "error": "Email already registered. Please log in.",
                "prefill": {
                    "name": name,
                    "email": email,
                    "gender": gender,
                    "age": age,
                    "height": height,
                },
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    user = User(
    name=name.strip(),
    email=email.lower().strip(),
    password_hash=pwd_context.hash(password),
    gender=gender.lower().strip(),   # ✅ force lowercase before saving
    age=age,
    height=height,
        )

    db.add(user)
    db.commit()
    db.refresh(user)

    request.session["user_email"] = user.email
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- LOGIN ----------
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user or not pwd_context.verify(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    request.session["user_email"] = user.email
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- DASHBOARD ----------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, category: str = None, db: Session = Depends(get_db)):
    """Show dashboard with basic info - optimized for fast loading."""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    # Fast basic data loading - no heavy operations
    try:
        # Quick wardrobe count for dashboard overview
        wardrobe_count = db.query(models.Outfit).filter(models.Outfit.user_id == user.id).count()
        
        # Get some recommended outfits for dashboard display
        recommended_outfits = db.query(models.Outfit).filter(models.Outfit.user_id == user.id).limit(4).all()
        
        # Basic weather data with short timeout
        weather_data = {"temp": 22, "condition": "Sunny", "city": "Islamabad"}
        try:
            city = "Islamabad"
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            resp = requests.get(url, timeout=3)  # Very short timeout
            if resp.ok:
                data = resp.json()
                weather_data = {
                    "temp": int(data.get("main", {}).get("temp", 22)),
                    "condition": data.get("weather", [{}])[0].get("main", "Sunny"),
                    "city": "Islamabad"
                }
        except Exception as e:
            logger.warning(f"Weather API timeout: {e}")
            # Use default weather data
            pass
        
        # Try to get today's AI outfit (generate if not exists)
        ai_outfit_url = None
        try:
            today = date.today().isoformat()
            outfit_path = os.path.join(OUTFIT_FOLDER, f"{today}.png")
            if os.path.exists(outfit_path):
                # Add cache-busting parameter based on file modification time
                file_mtime = int(os.path.getmtime(outfit_path))
                ai_outfit_url = f"/static/outfits/{today}.png?v={file_mtime}"
                logger.info(f"Using existing AI outfit for {today} with cache-busting")
            else:
                # Generate new AI outfit for today
                logger.info(f"Generating new AI outfit for {today}")
                ai_outfit_url = generate_outfit_of_the_day()
                if ai_outfit_url:
                    # Add cache-busting parameter for newly generated outfit
                    file_mtime = int(time.time())
                    ai_outfit_url = f"{ai_outfit_url}?v={file_mtime}"
                    logger.info(f"Successfully generated AI outfit: {ai_outfit_url}")
                else:
                    logger.warning("Failed to generate AI outfit")
        except Exception as e:
            logger.warning(f"AI outfit generation failed: {e}")
            pass
        
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user_email": user_email,
                "user": user,
                "wardrobe_count": wardrobe_count,
                "weather": weather_data,
                "ai_outfit_url": ai_outfit_url,
                "recommended_outfits": recommended_outfits,
                "combinations": [],  # Load via AJAX later if needed
                "wardrobe": {},  # Load via separate route
            },
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user_email": user_email,
                "user": user,
                "wardrobe_count": 0,
                "weather": {"temp": 22, "condition": "Sunny", "city": "Islamabad"},
                "ai_outfit_url": None,
                "recommended_outfits": [],
                "combinations": [],
                "wardrobe": {},
                "error": "Dashboard loading error - please refresh"
            },
        )

# ---------- LOGOUT ----------
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

# ---------- AI OUTFIT GENERATION ----------
@app.post("/generate-ai-outfit")
async def generate_ai_outfit_endpoint(request: Request, db: Session = Depends(get_db)):
    """Manually trigger AI outfit generation"""
    user_email = get_current_user_email(request)
    if not user_email:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    
    try:
        logger.info("Manual AI outfit generation triggered")
        # Force generation of a new outfit when user clicks the button
        ai_outfit_url = generate_outfit_of_the_day(force_new=True)
        if ai_outfit_url:
            # Add cache-busting parameter to ensure browser loads new image
            cache_buster = int(time.time())
            ai_outfit_url_with_cache = f"{ai_outfit_url}?v={cache_buster}"
            return JSONResponse({
                "success": True, 
                "outfit_url": ai_outfit_url_with_cache, 
                "message": "New AI outfit generated successfully!"
            })
        else:
            return JSONResponse({
                "success": False, 
                "message": "Failed to generate AI outfit. Please try again."
            })
    except Exception as e:
        logger.error(f"AI outfit generation error: {e}")
        return JSONResponse({
            "success": False, 
            "message": f"Error generating outfit: {str(e)}"
        })

# ---------- NEW TEMPLATE ROUTES ----------
@app.get("/weather-outfits", response_class=HTMLResponse)
async def weather_outfits(request: Request, db: Session = Depends(get_db)):
    """Weather-based outfit recommendations page"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get weather data (using existing weather logic)
    city = "Islamabad"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    weather_data = {"condition": "Sunny", "temperature": "25", "location": "London, UK"}
    
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json() if resp.ok else {}
        temp = float(data.get("main", {}).get("temp")) if data.get("main") else 25
        condition = (data.get("weather", [{}])[0].get("main") or "Sunny")
        weather_data = {
            "condition": condition,
            "temperature": str(int(temp)),
            "location": f"{city}, PK"
        }
    except Exception:
        pass
    
    # Get weather-appropriate outfits
    weather_outfits = []
    outfits_query = db.query(Outfit).filter(Outfit.user_id == user.id)
    if weather_data["temperature"].isdigit():
        temp = int(weather_data["temperature"])
        if temp < 15:
            # Cold weather outfits - broader categories
            weather_outfits = outfits_query.filter(
                or_(
                    Outfit.category.ilike("%jacket%"),
                    Outfit.category.ilike("%coat%"),
                    Outfit.category.ilike("%sweater%"),
                    Outfit.category.ilike("%hoodie%"),
                    Outfit.category.ilike("%blazer%"),
                    Outfit.category.ilike("%cardigan%"),
                    Outfit.subcategory.ilike("%jacket%"),
                    Outfit.subcategory.ilike("%coat%"),
                    Outfit.subcategory.ilike("%sweater%"),
                    Outfit.subcategory.ilike("%hoodie%"),
                    Outfit.subcategory.ilike("%blazer%"),
                    Outfit.subcategory.ilike("%cardigan%")
                )
            ).limit(4).all()
        elif temp > 30:
            # Very hot weather (30°C+) - prioritize cooling items
            weather_outfits = outfits_query.filter(
                or_(
                    Outfit.category.ilike("%shorts%"),
                    Outfit.category.ilike("%tank%"),
                    Outfit.category.ilike("%t-shirt%"),
                    Outfit.category.ilike("%dress%"),
                    Outfit.category.ilike("%skirt%"),
                    Outfit.category.ilike("%sandals%"),
                    Outfit.subcategory.ilike("%shorts%"),
                    Outfit.subcategory.ilike("%tank%"),
                    Outfit.subcategory.ilike("%t-shirt%"),
                    Outfit.subcategory.ilike("%dress%"),
                    Outfit.subcategory.ilike("%skirt%"),
                    Outfit.subcategory.ilike("%sandals%")
                )
            ).limit(4).all()
        elif temp > 25:
            # Hot weather outfits - include breathable items
            weather_outfits = outfits_query.filter(
                or_(
                    Outfit.category.ilike("%shorts%"),
                    Outfit.category.ilike("%t-shirt%"),
                    Outfit.category.ilike("%tank%"),
                    Outfit.category.ilike("%dress%"),
                    Outfit.category.ilike("%shirt%"),
                    Outfit.category.ilike("%blouse%"),
                    Outfit.category.ilike("%skirt%"),
                    Outfit.subcategory.ilike("%shorts%"),
                    Outfit.subcategory.ilike("%t-shirt%"),
                    Outfit.subcategory.ilike("%tank%"),
                    Outfit.subcategory.ilike("%dress%"),
                    Outfit.subcategory.ilike("%shirt%"),
                    Outfit.subcategory.ilike("%blouse%"),
                    Outfit.subcategory.ilike("%skirt%")
                )
            ).limit(4).all()
        elif temp > 20:
            # Mild weather outfits - comfortable layers
            weather_outfits = outfits_query.filter(
                or_(
                    Outfit.category.ilike("%jacket%"),
                    Outfit.category.ilike("%sweater%"),
                    Outfit.category.ilike("%hoodie%"),
                    Outfit.category.ilike("%pants%"),
                    Outfit.category.ilike("%jeans%"),
                    Outfit.category.ilike("%trousers%"),
                    Outfit.subcategory.ilike("%jacket%"),
                    Outfit.subcategory.ilike("%sweater%"),
                    Outfit.subcategory.ilike("%hoodie%"),
                    Outfit.subcategory.ilike("%pants%"),
                    Outfit.subcategory.ilike("%jeans%"),
                    Outfit.subcategory.ilike("%trousers%")
                )
            ).limit(4).all()
        else:
            # Cool weather (under 20°C) - add warm layers
            weather_outfits = outfits_query.filter(
                or_(
                    Outfit.category.ilike("%jacket%"),
                    Outfit.category.ilike("%sweater%"),
                    Outfit.category.ilike("%hoodie%"),
                    Outfit.category.ilike("%pants%"),
                    Outfit.category.ilike("%jeans%"),
                    Outfit.category.ilike("%trousers%"),
                    Outfit.subcategory.ilike("%jacket%"),
                    Outfit.subcategory.ilike("%sweater%"),
                    Outfit.subcategory.ilike("%hoodie%"),
                    Outfit.subcategory.ilike("%pants%"),
                    Outfit.subcategory.ilike("%jeans%"),
                    Outfit.subcategory.ilike("%trousers%")
                )
            ).limit(4).all()
            
    # If no weather-specific outfits found, show some random outfits
    if not weather_outfits:
        weather_outfits = outfits_query.limit(4).all()
    
    return templates.TemplateResponse(
        "weather_outfits.html",
        {
            "request": request,
            "user": user,
            "weather": weather_data,
            "weather_outfits": weather_outfits
        }
    )

@app.get("/api/weather/{city}")
async def get_weather_for_city(
    city: str,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get weather data for a specific city"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404
    
    # Get weather data for the specified city
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            temp = float(data.get("main", {}).get("temp", 25))
            condition = data.get("weather", [{}])[0].get("main", "Sunny")
            country = data.get("sys", {}).get("country", "")
            
            weather_data = {
                "condition": condition,
                "temperature": str(int(temp)),
                "location": f"{city.title()}, {country}"
            }
            
            # Get weather-appropriate outfits for this temperature
            weather_outfits = []
            outfits_query = db.query(Outfit).filter(Outfit.user_id == user.id)
            
            if temp > 30:
                # Very hot weather outfits (30°C+) - prioritize cooling items
                weather_outfits = outfits_query.filter(
                    Outfit.category.in_(["shorts", "tank-top", "t-shirt", "dress", "skirt", "sandals"])
                ).limit(8).all()
            elif temp > 25:
                # Hot weather outfits (25-30°C)
                weather_outfits = outfits_query.filter(
                    Outfit.category.in_(["t-shirt", "shorts", "dress", "tank-top", "shirt", "skirt"])
                ).limit(8).all()
            elif temp > 20:
                # Mild weather outfits (20-25°C)
                weather_outfits = outfits_query.filter(
                    Outfit.category.in_(["shirt", "jeans", "trousers", "pants", "chinos"])
                ).limit(8).all()
            elif temp > 15:
                # Cool weather outfits (15-20°C)
                weather_outfits = outfits_query.filter(
                    Outfit.category.in_(["jacket", "sweater", "hoodie", "pants", "jeans"])
                ).limit(8).all()
            else:
                # Cold weather outfits (under 15°C)
                weather_outfits = outfits_query.filter(
                    Outfit.category.in_(["jacket", "coat", "sweater", "hoodie", "blazer", "boots"])
                ).limit(8).all()
            
            # Format outfit data for JSON response
            outfit_data = []
            for outfit in weather_outfits:
                outfit_data.append({
                    "id": outfit.id,
                    "category": outfit.category,
                    "subcategory": outfit.subcategory,
                    "image_path": outfit.image_path,
                    "name": outfit.subcategory or outfit.category
                })
            
            return {
                "weather": weather_data,
                "outfits": outfit_data,
                "success": True
            }
        else:
            return {"error": "City not found", "success": False}, 404
    
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        return {"error": "Weather service unavailable", "success": False}, 500

@app.get("/outfit-combinations", response_class=HTMLResponse)
async def outfit_combinations_page(request: Request, db: Session = Depends(get_db)):
    """AI-generated outfit combinations page"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get user's wardrobe for combinations
    wardrobe = get_user_wardrobe(user, db)
    
    # Generate real AI outfit combinations from user's wardrobe
    outfit_combinations = []
    
    # Check if user has enough items for combinations
    total_items = sum(len(items) for items in wardrobe.values())
    
    if total_items >= 3:  # Need at least 3 items to make combinations
        try:
            # Use existing combination generation logic
            def generate_page_combinations(wardrobe, n=6):
                """Generate combinations for the combinations page"""
                wardrobe_items = []
                for category, outfits in wardrobe.items():
                    for outfit in outfits:
                        item_name = outfit.subcategory or outfit.category
                        wardrobe_items.append(item_name)
                
                if len(wardrobe_items) < 3:
                    return []
                
                # Configure Gemini
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    raise ValueError("GEMINI_API_KEY environment variable is required but not set.")
                genai.configure(api_key=gemini_api_key)
                gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                
                prompt = f"""
                Create {n} stylish outfit combinations from these wardrobe items: {wardrobe_items}
                
                Requirements:
                1. Use ONLY items from the provided wardrobe list
                2. Each combination needs: top, bottom, shoes
                3. Create cohesive, stylish combinations
                4. Vary the combinations for different occasions
                5. Only use items that actually exist in the wardrobe
                
                Respond ONLY in this exact JSON format:
                [
                  {{"top": "exact_item_name", "bottom": "exact_item_name", "shoes": "exact_item_name", "color_theme": "color description", "style_notes": "brief style description"}}
                ]
                """
                
                try:
                    response = gemini_model.generate_content(prompt)
                    text = response.text.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0]
                    elif "```" in text:
                        text = text.split("```")[1]
                    return json.loads(text.strip())
                except Exception as e:
                    logger.error(f"Page combination generation failed: {e}")
                    return []
            
            def map_page_combinations(user, db, combos):
                """Map generated combinations to actual outfit objects"""
                results = []
                for i, combo in enumerate(combos):
                    top_item = combo.get('top', '')
                    bottom_item = combo.get('bottom', '')
                    shoes_item = combo.get('shoes', '')
                    
                    # Find matching items in database
                    top = db.query(Outfit).filter(
                        Outfit.user_id == user.id,
                        or_(
                            Outfit.subcategory.ilike(f"%{top_item}%"),
                            Outfit.category.ilike(f"%{top_item}%")
                        )
                    ).first()
                    
                    bottom = db.query(Outfit).filter(
                        Outfit.user_id == user.id,
                        or_(
                            Outfit.subcategory.ilike(f"%{bottom_item}%"),
                            Outfit.category.ilike(f"%{bottom_item}%")
                        )
                    ).first()
                    
                    shoes = db.query(Outfit).filter(
                        Outfit.user_id == user.id,
                        or_(
                            Outfit.subcategory.ilike(f"%{shoes_item}%"),
                            Outfit.category.ilike(f"%{shoes_item}%")
                        )
                    ).first()
                    
                    if top and bottom and shoes:
                        results.append({
                            "id": f"combo_{i+1}",
                            "top": top,
                            "bottom": bottom,
                            "shoes": shoes,
                            "color_theme": combo.get('color_theme', 'Coordinated colors'),
                            "style_notes": combo.get('style_notes', 'Stylish combination')
                        })
                
                return results
            
            # Generate combinations
            combos_json = generate_page_combinations(wardrobe, n=6)
            outfit_combinations = map_page_combinations(user, db, combos_json)
            
        except Exception as e:
            logger.error(f"Error generating outfit combinations: {e}")
            outfit_combinations = []
    
    return templates.TemplateResponse(
        "outfit_combinations.html",
        {
            "request": request,
            "user": user,
            "outfit_combinations": outfit_combinations
        }
    )

@app.get("/style-assistant", response_class=HTMLResponse)
async def style_assistant_page(request: Request, db: Session = Depends(get_db)):
    """Style assistant chatbot page"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get chat history (if any)
    chat_history = []
    
    return templates.TemplateResponse(
        "style_assistant.html",
        {
            "request": request,
            "user": user,
            "chat_history": chat_history
        }
    )

@app.get("/calendar", response_class=HTMLResponse)
async def calendar_page(request: Request, db: Session = Depends(get_db)):
    """Calendar page for outfit planning"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # For now, redirect to dashboard as calendar is not yet implemented
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- UPLOAD OUTFIT ----------
@app.post("/upload_outfit")
async def upload_outfit(
    request: Request,
    category: str = Form(...),
    subcategory: str = Form(...),  # Made required
    outfit_image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload outfit image to user's wardrobe."""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    logger.info(f"Upload attempt - Filename: {outfit_image.filename}, Content Type: {outfit_image.content_type}")
    
    if not allowed_file(outfit_image.filename):
        logger.warning(f"Invalid file type rejected: {outfit_image.filename}")
        wardrobe = get_user_wardrobe(user, db)
        
        # Get weather data for the template
        city = "Islamabad"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        temp = None
        condition = "Unknown"
        try:
            resp = requests.get(url, timeout=15)
            data = resp.json() if resp.ok else {}
            temp = float(data.get("main", {}).get("temp")) if data.get("main") else None
            condition = (data.get("weather", [{}])[0].get("main") or "Unknown")
        except Exception:
            temp = None
            condition = "Unknown"
        
        # Get recommended outfits with improved temperature ranges
        outfits_query = db.query(models.Outfit).filter(models.Outfit.user_id == user.id)
        very_hot_cats = ["shorts", "tank-top", "t-shirt", "dress", "skirt", "sandals"]
        hot_cats = ["t-shirt", "shorts", "dress", "shirt", "skirt", "blouse"]
        mild_cats = ["shirt", "jeans", "trousers", "pants", "chinos"]
        cool_cats = ["jacket", "sweater", "hoodie", "pants", "jeans", "boots"]
        cold_cats = ["coat", "jacket", "sweater", "hoodie", "blazer", "boots", "pants", "trousers"]
        
        if temp is None:
            recommended_outfits = outfits_query.all()
        elif temp > 30:
            recommended_outfits = outfits_query.filter(models.Outfit.category.in_(very_hot_cats)).all()
        elif temp > 25:
            recommended_outfits = outfits_query.filter(models.Outfit.category.in_(hot_cats)).all()
        elif temp > 20:
            recommended_outfits = outfits_query.filter(models.Outfit.category.in_(mild_cats)).all()
        elif temp > 15:
            recommended_outfits = outfits_query.filter(models.Outfit.category.in_(cool_cats)).all()
        else:
            recommended_outfits = outfits_query.filter(models.Outfit.category.in_(cold_cats)).all()

        # Get combinations
        user_outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
        combinations = []
        for i in range(len(user_outfits)):
            for j in range(i + 1, len(user_outfits)):
                combinations.append((user_outfits[i], user_outfits[j]))
        
        response = templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "user_email": user_email,
                "user": user,
                "wardrobe": wardrobe,
                "selected_category": None,
                "ai_outfit_url": None,
                "weather": {"city": city, "temp": temp if temp is not None else "-", "condition": condition},
                "recommended_outfits": recommended_outfits,
                "combinations": combinations,
                "error": "Invalid file type. Allowed: png, jpg, jpeg, gif, webp, bmp, tiff.",
            },
        )
        return response

    filename = secure_filename(outfit_image.filename)
    logger.info(f"Processing file: {filename}")
    
    user_folder = os.path.join(UPLOAD_FOLDER, str(user.id))
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)
    logger.info(f"Saving to: {file_path}")

    if os.path.exists(file_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        file_path = os.path.join(user_folder, filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await outfit_image.read()
            buffer.write(content)
        logger.info(f"File saved successfully: {file_path} ({len(content)} bytes)")
        
        # Detect colors after saving the file
        logger.info("Starting color detection...")
        try:
            detected_colors = detect_image_colors(file_path, max_colors=3)
            primary_color = detected_colors[0]['color_name'] if detected_colors else 'unknown'
            secondary_color = detected_colors[1]['color_name'] if len(detected_colors) > 1 else None

            outfit = Outfit(
                user_id=user.id,
                category=category,
                subcategory=subcategory,
                image_path=os.path.join("uploads", str(user.id), filename),
                primary_color=primary_color,
                secondary_color=secondary_color,
            )
            db.add(outfit)
            db.commit()
            db.refresh(outfit)

            wardrobe = get_user_wardrobe(user, db)

            # Get weather data for the template
            city = "Islamabad"
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
            temp = None
            condition = "Unknown"
            try:
                resp = requests.get(url, timeout=15)
                data = resp.json() if resp.ok else {}
                temp = float(data.get("main", {}).get("temp")) if data.get("main") else None
                condition = (data.get("weather", [{}])[0].get("main") or "Unknown")
            except Exception:
                temp = None
                condition = "Unknown"
            
            # Get recommended outfits with improved temperature ranges
            outfits_query = db.query(models.Outfit).filter(models.Outfit.user_id == user.id)
            very_hot_cats = ["shorts", "tank-top", "t-shirt", "dress", "skirt", "sandals"]
            hot_cats = ["t-shirt", "shorts", "dress", "shirt", "skirt", "blouse"]
            mild_cats = ["shirt", "jeans", "trousers", "pants", "chinos"]
            cool_cats = ["jacket", "sweater", "hoodie", "pants", "jeans", "boots"]
            cold_cats = ["coat", "jacket", "sweater", "hoodie", "blazer", "boots", "pants", "trousers"]
            
            if temp is None:
                recommended_outfits = outfits_query.all()
            elif temp > 30:
                recommended_outfits = outfits_query.filter(models.Outfit.category.in_(very_hot_cats)).all()
            elif temp > 25:
                recommended_outfits = outfits_query.filter(models.Outfit.category.in_(hot_cats)).all()
            elif temp > 20:
                recommended_outfits = outfits_query.filter(models.Outfit.category.in_(mild_cats)).all()
            elif temp > 15:
                recommended_outfits = outfits_query.filter(models.Outfit.category.in_(cool_cats)).all()
            else:
                recommended_outfits = outfits_query.filter(models.Outfit.category.in_(cold_cats)).all()

            # Get combinations
            user_outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
            combinations = []
            for i in range(len(user_outfits)):
                for j in range(i + 1, len(user_outfits)):
                    combinations.append((user_outfits[i], user_outfits[j]))
            
            response = templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "user_email": user_email,
                    "user": user,
                    "wardrobe": wardrobe,
                    "selected_category": None,
                    "ai_outfit_url": None,
                    "weather": {"city": city, "temp": temp if temp is not None else "-", "condition": condition},
                    "recommended_outfits": recommended_outfits,
                    "combinations": combinations,
                },
            )
            return response

        except Exception as e:
            logger.error(f"Error detecting colors: {e}")
            return JSONResponse(
                content={"error": "Failed to detect colors"},
                status_code=500
            )

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return JSONResponse(
            content={"error": "Failed to save file"},
            status_code=500
        )

@app.post("/upload-model-image")
async def upload_model_image(
    request: Request,
    model_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a model image for try-on"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    if not allowed_file(model_image.filename):
        return JSONResponse(
            content={"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, webp, bmp, tiff."},
            status_code=400
        )

    filename = secure_filename(model_image.filename)
    user_folder = os.path.join(MODEL_IMAGES_FOLDER, str(user.id))
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, filename)

    if os.path.exists(file_path):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{int(time.time())}{ext}"
        file_path = os.path.join(user_folder, filename)

    try:
        with open(file_path, "wb") as buffer:
            content = await model_image.read()
            buffer.write(content)
        
        rel_path = os.path.join("model_images", str(user.id), filename)
        
        model_img = ModelImage(
            user_id=user.id,
            image_path=rel_path
        )
        db.add(model_img)
        db.commit()
        
        return JSONResponse(content={"success": True, "image_id": model_img.id})
        
    except Exception as e:
        logger.error(f"Error saving model image: {e}")
        return JSONResponse(
            content={"error": "Failed to save model image"},
            status_code=500
        )

    # Color detection should be part of the upload_outfit function, not here
    # This code seems to be misplaced from another function
    # Let's move it to the correct place in upload_outfit

# This part appears to be misplaced code from upload_outfit function - should be moved there
# color_confidence = detected_colors[0]['confidence'] if detected_colors else 0
# 
# # Store detailed color data as JSON
# color_data = json.dumps(detected_colors) if detected_colors else None
# 
# logger.info(f"Color detection completed. Primary: {primary_color}, Secondary: {secondary_color}, Confidence: {color_confidence}%")

# except Exception as color_error:
#     logger.error(f"Color detection failed: {color_error}")
#     primary_color = 'unknown'
#     secondary_color = None
#     color_confidence = 0
#     color_data = None
            
# except Exception as e:
#     logger.error(f"Error saving file: {e}")
#     return templates.TemplateResponse(
#         "dashboard.html",
#         {
#             "request": request,
#             "user_email": user_email,
#             "user": user,
#             "wardrobe": get_user_wardrobe(user, db),
#             "error": f"Error saving file: {str(e)}",
#         },
#     )

# rel_path = os.path.join("uploads", str(user.id), filename)
# # normalize category for consistency
# normalized_category = (category or "").strip().lower()
# # subcategory is now required, but handle edge cases
# normalized_subcategory = (subcategory or "").strip().lower() if subcategory and subcategory.strip() else None
# 
# # Validate that subcategory is provided
# if not normalized_subcategory:
#     logger.error("Subcategory is required but was not provided")
#     return templates.TemplateResponse(
#         "upload_outfit.html",
#         {
#             "request": request,
#             "user_email": user_email,
#             "user": user,
#             "error": "Subcategory is required. Please select or add a subcategory.",
#         },
#     )
# 
# outfit = Outfit(
#     user_id=user.id, 
#     category=normalized_category, 
#     subcategory=normalized_subcategory,
#     image_path=rel_path,
#     primary_color=primary_color,
#     secondary_color=secondary_color,
#     color_confidence=int(color_confidence) if color_confidence else None,
#     color_data=color_data
# )
# db.add(outfit)
# db.commit()

# return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- DELETE OUTFIT ----------
@app.delete("/delete_outfit/{outfit_id}")
async def delete_outfit(
    outfit_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    """Delete an outfit from user's wardrobe."""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    # Find the outfit and verify it belongs to the user
    outfit = db.query(Outfit).filter(
        Outfit.id == outfit_id,
        Outfit.user_id == user.id
    ).first()
    
    if not outfit:
        return {"error": "Outfit not found"}, 404

    # Delete the image file if it exists
    try:
        image_path = os.path.join(BASE_DIR, "static", outfit.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        logger.warning(f"Could not delete image file: {e}")

    # Delete from database
    db.delete(outfit)
    db.commit()

    return {"message": "Outfit deleted successfully"}

# ---------- SEE WARDROBE ----------
@app.get("/see_wardrobe", response_class=HTMLResponse)
async def see_wardrobe(request: Request, category: str = None, db: Session = Depends(get_db)):
    """Direct route for viewing wardrobe section."""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    wardrobe = get_user_wardrobe(user, db, category)
    
    return templates.TemplateResponse(
        "wardrobe.html",
        {
            "request": request,
            "user_email": user_email,
            "user": user,
            "wardrobe": wardrobe,
            "selected_category": category,
        },
    )

# ---------- UPLOAD OUTFIT PAGE ----------
@app.get("/upload_outfit", response_class=HTMLResponse)
async def upload_outfit_page(request: Request, db: Session = Depends(get_db)):
    """Show upload outfit page."""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        "upload_outfit.html",
        {
            "request": request,
            "user_email": user_email,
            "user": user,
        },
    )

# ---------- NEW ROUTES FOR CONSISTENT NAVIGATION ----------
@app.get("/wardrobe", response_class=HTMLResponse)
async def wardrobe(request: Request, category: str = None, db: Session = Depends(get_db)):
    """Route for /wardrobe - redirects to see_wardrobe functionality."""
    return await see_wardrobe(request, category, db)

@app.get("/upload-outfit", response_class=HTMLResponse)
async def upload_outfit_dash(request: Request, db: Session = Depends(get_db)):
    """Route for /upload-outfit - redirects to upload_outfit functionality."""
    return await upload_outfit_page(request, db)

# ----------- weather -------------------
# inside app/main.py

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY environment variable is required but not set.")

@app.get("/recommendations")
async def recommendations_redirect():
    # Redirect old route to integrated dashboard view
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- COMBINATIONS ---------
@app.get("/combinations")
async def combinations_redirect():
    # Redirect old route to integrated dashboard view
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)

# ---------- LOAD MORE COMBINATIONS API ---------
@app.get("/api/more_combinations")
async def get_more_combinations(
    request: Request,
    current_count: int = 0,
    db: Session = Depends(get_db)
):
    """API endpoint to get more combinations without full page reload"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    # Get user's wardrobe
    wardrobe = get_user_wardrobe(user, db)
    
    # Configure Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required but not set.")
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_new_combinations(wardrobe, offset: int = 0):
        wardrobe_items = []
        for category, outfits in wardrobe.items():
            for outfit in outfits:
                item_name = outfit.subcategory or outfit.category
                wardrobe_items.append(item_name)
        
        if len(wardrobe_items) < 3:
            return []
        
        prompt = f"""
        You are an expert fashion stylist. Create 5 NEW outfit combinations from these wardrobe items:
        {wardrobe_items}
        
        Requirements:
        - Each combination needs: top + bottom + shoes
        - Focus on color coordination and style harmony
        - Make combinations #{offset + 1} to #{offset + 5} (different from previous suggestions)
        - Only use items that exist in the wardrobe
        
        JSON format only:
        [
          {{"top": "item_name", "bottom": "item_name", "shoes": "item_name", "color_theme": "theme", "style_notes": "notes"}}
        ]
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"API combination generation failed: {e}")
            return []

    def map_to_outfits(user, db, combos):
        results = []
        for combo in combos:
            top_item = combo.get('top', '')
            bottom_item = combo.get('bottom', '')
            shoes_item = combo.get('shoes', '')
            
            top = db.query(Outfit).filter(
                Outfit.user_id == user.id,
                (Outfit.subcategory.ilike(f"%{top_item}%") | Outfit.category.ilike(f"%{top_item}%"))
            ).first()
            
            bottom = db.query(Outfit).filter(
                Outfit.user_id == user.id,
                (Outfit.subcategory.ilike(f"%{bottom_item}%") | Outfit.category.ilike(f"%{bottom_item}%"))
            ).first()
            
            shoes = db.query(Outfit).filter(
                Outfit.user_id == user.id,
                (Outfit.subcategory.ilike(f"%{shoes_item}%") | Outfit.category.ilike(f"%{shoes_item}%"))
            ).first()
            
            if top and bottom and shoes:
                results.append({
                    "top": {
                        "id": top.id,
                        "image_path": top.image_path,
                        "category": top.category,
                        "subcategory": top.subcategory
                    },
                    "bottom": {
                        "id": bottom.id,
                        "image_path": bottom.image_path,
                        "category": bottom.category,
                        "subcategory": bottom.subcategory
                    },
                    "shoes": {
                        "id": shoes.id,
                        "image_path": shoes.image_path,
                        "category": shoes.category,
                        "subcategory": shoes.subcategory
                    },
                    "color_theme": combo.get('color_theme', 'Coordinated colors'),
                    "style_notes": combo.get('style_notes', 'Stylish combination')
                })
        return results

    # Generate new combinations
    new_combos = generate_new_combinations(wardrobe, offset=current_count)
    combinations = map_to_outfits(user, db, new_combos)
    
    return {
        "combinations": combinations,
        "total_count": current_count + len(combinations),
        "new_count": len(combinations)
    }


# ---------- COLOR-BASED SEARCH AND FILTERING ----------
@app.get("/api/search_by_color")
async def search_by_color(
    request: Request,
    color: str,
    db: Session = Depends(get_db)
):
    """Search outfits by color name"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    # Search by primary or secondary color
    outfits = db.query(Outfit).filter(
        Outfit.user_id == user.id,
        (Outfit.primary_color.ilike(f"%{color}%") | 
         Outfit.secondary_color.ilike(f"%{color}%"))
    ).all()

    outfit_data = []
    for outfit in outfits:
        outfit_data.append({
            "id": outfit.id,
            "category": outfit.category,
            "subcategory": outfit.subcategory,
            "image_path": outfit.image_path,
            "primary_color": outfit.primary_color,
            "secondary_color": outfit.secondary_color,
            "color_confidence": outfit.color_confidence
        })

    return {"outfits": outfit_data, "count": len(outfit_data)}

@app.get("/api/outfit_colors/{outfit_id}")
async def get_outfit_colors(
    outfit_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Get detailed color information for a specific outfit"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    outfit = db.query(Outfit).filter(
        Outfit.id == outfit_id,
        Outfit.user_id == user.id
    ).first()

    if not outfit:
        return {"error": "Outfit not found"}, 404

    # Parse color data if available
    detailed_colors = []
    if outfit.color_data:
        try:
            detailed_colors = json.loads(outfit.color_data)
        except json.JSONDecodeError:
            pass

    return {
        "outfit_id": outfit.id,
        "primary_color": outfit.primary_color,
        "secondary_color": outfit.secondary_color,
        "color_confidence": outfit.color_confidence,
        "detailed_colors": detailed_colors
    }

@app.post("/api/redetect_colors/{outfit_id}")
async def redetect_outfit_colors(
    outfit_id: int,
    request: Request,
    db: Session = Depends(get_db)
):
    """Re-run color detection on an existing outfit"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    outfit = db.query(Outfit).filter(
        Outfit.id == outfit_id,
        Outfit.user_id == user.id
    ).first()

    if not outfit:
        return {"error": "Outfit not found"}, 404

    # Get full image path
    image_path = os.path.join(BASE_DIR, "static", outfit.image_path)
    
    if not os.path.exists(image_path):
        return {"error": "Image file not found"}, 404

    try:
        # Re-detect colors
        detected_colors = detect_image_colors(image_path, max_colors=3)
        primary_color = detected_colors[0]['color_name'] if detected_colors else 'unknown'
        secondary_color = detected_colors[1]['color_name'] if len(detected_colors) > 1 else None
        color_confidence = detected_colors[0]['confidence'] if detected_colors else 0
        color_data = json.dumps(detected_colors) if detected_colors else None

        # Update database
        outfit.primary_color = primary_color
        outfit.secondary_color = secondary_color
        outfit.color_confidence = int(color_confidence) if color_confidence else None
        outfit.color_data = color_data
        db.commit()

        return {
            "success": True,
            "primary_color": primary_color,
            "secondary_color": secondary_color,
            "color_confidence": color_confidence,
            "detailed_colors": detected_colors
        }

    except Exception as e:
        logger.error(f"Error re-detecting colors: {e}")
        return {"error": f"Color detection failed: {str(e)}"}, 500

@app.get("/api/color_statistics")
async def get_color_statistics(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get color distribution statistics for user's wardrobe"""
    user_email = get_current_user_email(request)
    if not user_email:
        return {"error": "Not authenticated"}, 401

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return {"error": "User not found"}, 404

    # Get all outfits with colors
    outfits = db.query(Outfit).filter(
        Outfit.user_id == user.id,
        Outfit.primary_color.isnot(None)
    ).all()

    # Count colors
    color_count = {}
    for outfit in outfits:
        if outfit.primary_color:
            color_count[outfit.primary_color] = color_count.get(outfit.primary_color, 0) + 1
        if outfit.secondary_color:
            color_count[outfit.secondary_color] = color_count.get(outfit.secondary_color, 0) + 1

    # Sort by frequency
    sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)

    return {
        "total_items": len(outfits),
        "color_distribution": sorted_colors,
        "most_common_color": sorted_colors[0][0] if sorted_colors else None,
        "unique_colors": len(color_count)
    }


# ============== SIMPLE CHAT API ==============

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict] = None

def get_current_user(request: Request, db: Session = Depends(get_db)):
    """Get current user from session"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.get("/api/wardrobe-items")
async def get_wardrobe_items(request: Request, db: Session = Depends(get_db)):
    """Get all wardrobe items for the current user"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get all wardrobe items
    outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
    
    items = []
    for outfit in outfits:
        items.append({
            "id": outfit.id,
            "category": outfit.category,
            "subcategory": outfit.subcategory,
            "image_path": outfit.image_path,
            "primary_color": outfit.primary_color,
            "secondary_color": outfit.secondary_color
        })
    
    return {"items": items}

@app.get("/api/chat-history")
async def get_chat_history(limit: int = 10, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Get user's recent chat history"""
    try:
        from app.chatbot_service import chatbot
        history = chatbot.get_chat_history(current_user, db, limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return {"history": []}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Enhanced chat endpoint with wardrobe item context and web search integration"""
    try:
        response_data = await chatbot.process_message(
            user=current_user,
            message=request.message,
            db=db,
            context=request.context
        )
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return JSONResponse(
            content={
                "response": "I'm experiencing some technical difficulties. Please try again in a moment!",
                "type": "error",
                "images": [],
                "suggestions": []
            },
            status_code=500
        )

@app.get("/api/chat/wardrobe-analysis")
async def analyze_wardrobe(
    request: Request,
    db: Session = Depends(get_db)
):
    """Analyze user's wardrobe and suggest improvements"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        analysis = await chatbot.find_missing_wardrobe_items(user, db)
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Error analyzing wardrobe: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ---------- TRY ON FEATURE ----------

STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
if not STABILITY_API_KEY:
    raise ValueError("STABILITY_API_KEY environment variable is required but not set.")

class TryOnRequest(BaseModel):
    model_image_id: int
    outfit_ids: list[int]

@app.get("/try-on", response_class=HTMLResponse)
async def try_on_page(request: Request, db: Session = Depends(get_db)):
    """Display the try-on page where users can upload model images and select outfits"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get user's model images
    # Get user's model images
    model_images = db.query(ModelImage).filter(ModelImage.user_id == user.id).all()
    
    # Get user's wardrobe
    wardrobe = get_user_wardrobe(user, db)
    
    return templates.TemplateResponse(
        "try_on.html",
        {
            "request": request,
            "user": user,
            "model_images": model_images,
            "wardrobe": wardrobe
        }
    )

@app.post("/try-on-process")
async def try_on_process(
    request: Request,
    tryon_request: TryOnRequest,
    db: Session = Depends(get_db)
):
    """Process the try-on request using Gemini and Stability AI"""
    user_email = get_current_user_email(request)
    if not user_email:
        return JSONResponse(content={"error": "Not authenticated"}, status_code=401)

    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return JSONResponse(content={"error": "User not found"}, status_code=404)

    # Get model image
    model_image = db.query(ModelImage).filter(
        ModelImage.id == tryon_request.model_image_id,
        ModelImage.user_id == user.id
    ).first()
    
    if not model_image:
        return JSONResponse(content={"error": "Model image not found"}, status_code=404)

    # Get selected outfits
    outfits = db.query(Outfit).filter(
        Outfit.id.in_(tryon_request.outfit_ids),
        Outfit.user_id == user.id
    ).all()
    
    if not outfits:
        return JSONResponse(content={"error": "No outfits selected"}, status_code=400)

    try:
        # Prepare image paths
        model_image_path = os.path.join(BASE_DIR, "static", model_image.image_path)
        
        # Verify model image exists
        if not os.path.exists(model_image_path):
            return JSONResponse(content={"error": "Model image file not found"}, status_code=404)
        
        # Prepare outfit image paths
        outfit_image_paths = []
        for outfit in outfits:
            outfit_path = os.path.join(BASE_DIR, "static", outfit.image_path)
            if os.path.exists(outfit_path):
                outfit_image_paths.append(outfit_path)
        
        if not outfit_image_paths:
            return JSONResponse(content={"error": "No valid outfit images found"}, status_code=400)
        
        # Generate prompt using Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required but not set.")
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Create prompt for Gemini
        outfit_descriptions = [f"{outfit.category} ({outfit.subcategory})" for outfit in outfits]
        outfit_list = ", ".join(outfit_descriptions)
        
        prompt_request = f"""
You are a professional fashion AI assistant specializing in virtual try-on technology.

You will receive:
1. A MODEL image (a person).
2. One or more CLOTH images ({outfit_list}).

Your task:
- Analyze both images carefully.
- Generate **ONE single combined prompt** ONLY.
- This prompt must describe the MODEL as if he is realistically WEARING ALL the CLOTHES.
- Do NOT give separate descriptions.
- The prompt must be photorealistic, ultra-detailed, styled as a professional fashion photoshoot.
- Include details of lighting (studio lighting, soft shadows), texture (fabric weave, material quality), realism, natural skin tones, and accurate cloth fit.
- Describe how each clothing item fits the model's body shape and size.
- Include specific details about colors, patterns, and styling of each clothing item.
- The final image should have no background (transparent background).
- The final image should be 3D and realistic with professional photography quality.
- Ensure all selected clothes are properly worn by the model in a realistic way.
- Focus on creating a lifelike representation where the clothes appear naturally worn.
- Final output should be a single detailed prompt optimized for Stable Diffusion.
"""

        # Read model image
        with open(model_image_path, "rb") as f:
            model_image_data = f.read()
        
        # Read all outfit images and send them to Gemini
        gemini_content = [
            prompt_request,
            {"mime_type": "image/jpeg", "data": model_image_data}
        ]
        
        # Add all outfit images to the content
        for outfit_path in outfit_image_paths:
            with open(outfit_path, "rb") as f:
                outfit_image_data = f.read()
            gemini_content.append({"mime_type": "image/jpeg", "data": outfit_image_data})
        
        # Generate prompt with Gemini
        response = gemini_model.generate_content(gemini_content)
        
        final_prompt = response.text.strip()
        logger.info(f"Generated prompt: {final_prompt}")
        
        # Send to Stability AI
        url = "https://api.stability.ai/v2beta/stable-image/generate/core"
        
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "application/json"
        }
        
        # Stability AI requires proper multipart/form-data with specific field names
        data = {
            "prompt": (None, final_prompt),
            "output_format": (None, "png"),
            "negative_prompt": (None, "blurry, low quality, bad anatomy, malformed limbs, ugly, deformed, extra limbs, bad hands, bad eyes")
        }
        
        stability_response = requests.post(url, headers=headers, files=data)
        
        if stability_response.status_code == 200:
            stability_data = stability_response.json()
            img_base64 = stability_data["image"]
            img_bytes = base64.b64decode(img_base64)
            
            # Save the result
            result_filename = f"tryon_{user.id}_{int(time.time())}.png"
            result_path = os.path.join(TRYON_RESULTS_FOLDER, result_filename)
            
            with open(result_path, "wb") as f:
                f.write(img_bytes)
            
            result_url = f"/static/tryon_results/{result_filename}"
            
            return JSONResponse(content={
                "success": True,
                "result_url": result_url,
                "prompt": final_prompt
            })
        else:
            logger.error(f"Stability AI error: {stability_response.status_code} - {stability_response.text}")
            return JSONResponse(content={
                "error": f"Failed to generate image: {stability_response.text}"
            }, status_code=500)
            
    except Exception as e:
        logger.error(f"Error in try-on process: {e}")
        return JSONResponse(content={"error": f"Try-on process failed: {str(e)}"}, status_code=500)

@app.get("/see-model", response_class=HTMLResponse)
async def see_model_page(request: Request, db: Session = Depends(get_db)):
    """Display the see model page where users can view their try-on results"""
    user_email = get_current_user_email(request)
    if not user_email:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    
    # Get query parameters
    result_url = request.query_params.get('result')
    prompt = request.query_params.get('prompt')
    
    return templates.TemplateResponse(
        "see_model.html",
        {
            "request": request,
            "user": user,
            "result_url": result_url,
            "prompt": prompt
        }
    )

# ============== DETAILED CHATBOT API ENDPOINTS ==============

class ChatMessageRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    type: str
    images: list = []
    suggestions: list = []

@app.post("/api/chat/send", response_model=ChatResponse)
async def send_chat_message(
    request: Request,
    chat_request: ChatMessageRequest,
    db: Session = Depends(get_db)
):
    """Send message to chatbot and get response"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        response_data = await chatbot.process_message(
            user=user,
            message=chat_request.message,
            db=db,
            context=chat_request.context
        )
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chat/history")
async def get_chat_history(
    request: Request,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get user's chat history"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        history = chatbot.get_chat_history(user, db, limit)
        return {"history": history}
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat/outfit-advice")
async def get_outfit_advice(
    request: Request,
    outfit_id: int,
    question: str = Form(...),
    db: Session = Depends(get_db)
):
    """Get specific advice about an outfit item"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Verify outfit belongs to user
    outfit = db.query(Outfit).filter(
        Outfit.id == outfit_id,
        Outfit.user_id == user.id
    ).first()
    
    if not outfit:
        raise HTTPException(status_code=404, detail="Outfit not found")
    
    try:
        context = {"selected_item_id": outfit_id}
        response_data = await chatbot.process_message(
            user=user,
            message=question,
            db=db,
            context=context
        )
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error getting outfit advice: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chat/trends")
async def get_fashion_trends(
    request: Request,
    query: str = "latest fashion trends",
    db: Session = Depends(get_db)
):
    """Get current fashion trends"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        trends = await chatbot.get_fashion_trends(query)
        return {"trends": trends}
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/chat/wardrobe-analysis")
async def analyze_wardrobe(
    request: Request,
    db: Session = Depends(get_db)
):
    """Analyze user's wardrobe and suggest improvements"""
    user_email = get_current_user_email(request)
    if not user_email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        analysis = await chatbot.find_missing_wardrobe_items(user, db)
        return {"analysis": analysis}
        
    except Exception as e:
        logger.error(f"Error analyzing wardrobe: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


