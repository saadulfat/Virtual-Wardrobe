import os
import json
import logging
import aiohttp
import google.generativeai as genai
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from app.models import User, Outfit, ChatMessage
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class WardrobeChatbot:
    def __init__(self):
        # Initialize Gemini AI
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required but not set.")
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Google Search API configuration (SerpAPI)
        google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")  # Fallback to default if not set
        self.google_endpoint = "https://serpapi.com/search"
        self.google_api_key = google_api_key
        
        # Fashion and style context
        self.fashion_context = """
        You are StyleWise AI, the world's most advanced virtual fashion stylist and wardrobe consultant. You have complete access to the user's personal wardrobe inventory and provide expert fashion guidance with deep knowledge of trends, colors, and styling principles.
        
        YOUR CORE EXPERTISE:
        • WARDROBE MASTERY: Analyze user's clothing inventory, count items by category, identify color patterns
        • TREND AUTHORITY: Current 2024-2025 fashion trends, seasonal must-haves, emerging styles
        • STYLING GENIUS: Create perfect outfit combinations from existing wardrobe pieces
        • COLOR EXPERT: Advanced color theory, trending palettes, coordination techniques
        • SHOPPING ADVISOR: Identify wardrobe gaps, recommend essential additions
        • OCCASION SPECIALIST: Styling for work, casual, formal, party, travel, special events
        • BODY CONFIDENCE: Flattering fits, proportion enhancement, personal style development
        
        CURRENT FASHION TRENDS 2024-2025:
        • SILHOUETTES: Oversized blazers, wide-leg trousers, cropped jackets, maxi dresses, cargo details
        • STYLES: Dopamine dressing, dark academia, coastal grandmother, cottagecore revival
        • TEXTURES: Bouclé, corduroy, faux leather, sheer layers, metallic finishes, textured knits
        • DETAILS: Statement collars, dramatic sleeves, cutout designs, architectural shapes
        • SUSTAINABLE: Vintage revival, upcycled pieces, rental fashion, capsule wardrobes
        
        TRENDING COLORS 2024-2025:
        • PANTONE STARS: Peach Fuzz (#FFBE98), Mocha Mousse (#AD8B73), Digital Lime (#BFFF00)
        • EARTH TONES: Terracotta, sage green, warm camel, mushroom, dusty rose
        • BOLD STATEMENTS: Electric cobalt, sunset coral, deep emerald, royal purple, cherry red
        • SOPHISTICATED NEUTRALS: Oat milk, charcoal, cream, soft black, warm grey
        • METALLICS: Champagne gold, rose gold, gunmetal, copper, brushed silver
        
        RESPONSE STYLE:
        • Be enthusiastic, friendly, and encouraging about fashion
        • Keep responses concise (under 400 words) and well-formatted
        • Use short paragraphs (2-3 sentences max) for better readability
        • Reference user's actual wardrobe items by category and description
        • Include trend context and styling tips
        • Use emojis for visual appeal but avoid excessive markdown
        • Give multiple styling options when possible
        • Be inclusive, body-positive, and confidence-building
        • Answer questions directly and specifically
        • When counting wardrobe items, provide exact numbers and brief descriptions
        • Break long responses into digestible chunks
        """

    async def get_user_wardrobe_context(self, user: User, db: Session) -> str:
        """Get detailed wardrobe context for the AI"""
        outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
        
        if not outfits:
            return "User's wardrobe is currently empty. Encourage them to upload some clothing items first."
        
        wardrobe_summary = {
            "total_items": len(outfits),
            "categories": {},
            "items": [],
            "detailed_inventory": []
        }
        
        for outfit in outfits:
            category = outfit.category.lower()
            subcategory = (outfit.subcategory or "general").lower()
            
            if category not in wardrobe_summary["categories"]:
                wardrobe_summary["categories"][category] = []
            
            wardrobe_summary["categories"][category].append(subcategory)
            wardrobe_summary["items"].append({
                "id": outfit.id,
                "category": category,
                "subcategory": subcategory,
                "image_path": outfit.image_path
            })
            
            # Create detailed inventory for AI
            wardrobe_summary["detailed_inventory"].append(f"{subcategory} ({category})")
        
        # Create a comprehensive wardrobe description
        wardrobe_description = f"User's Complete Wardrobe Inventory ({wardrobe_summary['total_items']} items):\n\n"
        
        for category, subcategories in wardrobe_summary["categories"].items():
            unique_subcategories = list(set(subcategories))
            count = len([item for item in wardrobe_summary["items"] if item["category"] == category])
            wardrobe_description += f"• {category.title()} ({count} items): {', '.join(unique_subcategories)}\n"
        
        wardrobe_description += f"\nDetailed Item List: {', '.join(wardrobe_summary['detailed_inventory'])}"
        
        return wardrobe_description

    async def search_wardrobe_items(self, message: str, user_id: int) -> str:
        """Search for specific items in user's wardrobe and return formatted response"""
        from app.database import SessionLocal
        db = SessionLocal()
        
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return "User not found."
            
            # Extract search terms from message
            message_lower = message.lower()
            search_terms = []
            
            # Common clothing categories and items
            categories = ['shirt', 'shirts', 'pants', 'jeans', 'dress', 'dresses', 
                         'jacket', 'jackets', 'shoes', 'boots', 'sweater', 'sweaters',
                         'hoodie', 'hoodies', 'blouse', 'blouses', 'skirt', 'skirts',
                         'shorts', 'sneakers', 't-shirt', 't-shirts']
            
            colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 
                     'purple', 'brown', 'grey', 'gray', 'navy', 'beige', 'orange']
            
            # Find categories mentioned in message
            for category in categories:
                if category in message_lower:
                    search_terms.append(category.rstrip('s'))  # Remove plural
            
            # Find colors mentioned in message
            for color in colors:
                if color in message_lower:
                    search_terms.append(color)
            
            if not search_terms:
                # If no specific terms found, show general wardrobe overview
                wardrobe_context = await self.get_user_wardrobe_context(user, db)
                return wardrobe_context
            
            # Search for items matching the terms
            outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
            matching_items = []
            
            for outfit in outfits:
                category = (outfit.category or "").lower()
                subcategory = (outfit.subcategory or "").lower()
                primary_color = (outfit.primary_color or "").lower()
                
                # Check if any search term matches
                for term in search_terms:
                    if (term in category or term in subcategory or term in primary_color):
                        matching_items.append(outfit)
                        break
            
            if matching_items:
                # Group by category for better presentation
                category_counts = {}
                subcategory_counts = {}
                
                for item in matching_items:
                    cat = item.category.lower()
                    subcat = (item.subcategory or item.category).lower()
                    
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1
                
                # Create response
                total_count = len(matching_items)
                
                if len(subcategory_counts) > 1:
                    # Multiple subcategories - show breakdown
                    subcategory_breakdown = []
                    for subcat, count in subcategory_counts.items():
                        display_name = subcat.replace('-', ' ')
                        subcategory_breakdown.append(f"{count} {display_name}{'s' if count > 1 else ''}")
                    breakdown_text = ", ".join(subcategory_breakdown)
                    
                    # Determine main category for the response
                    main_category = max(category_counts.items(), key=lambda x: x[1])[0]
                    return f"🧥 You have {total_count} {main_category}{'s' if total_count > 1 else ''}:\n\n{breakdown_text}"
                else:
                    # Single subcategory type
                    subcat_name = list(subcategory_counts.keys())[0].replace('-', ' ')
                    return f"🧥 You have {total_count} {subcat_name}{'s' if total_count > 1 else ''}"
            else:
                search_term = ' '.join(search_terms)
                return f"🚫 No {search_term} items found in your wardrobe. Consider adding some!"
                
        except Exception as e:
            logger.error(f"Error searching wardrobe items: {e}")
            return "I had trouble searching your wardrobe. Please try again!"
        finally:
            db.close()

    async def analyze_outfit_combination(self, user: User, db: Session, selected_item_id: Optional[int] = None) -> str:
        """Analyze outfit combinations from user's wardrobe"""
        outfits = db.query(Outfit).filter(Outfit.user_id == user.id).all()
        
        if len(outfits) < 2:
            return "You need at least 2 items in your wardrobe to get combination suggestions. Try uploading more clothes!"
        
        selected_item = None
        if selected_item_id:
            selected_item = db.query(Outfit).filter(
                Outfit.id == selected_item_id, 
                Outfit.user_id == user.id
            ).first()
        
        # Create item descriptions for AI
        item_descriptions = []
        for outfit in outfits:
            item_descriptions.append(f"{outfit.category}/{outfit.subcategory or 'general'}")
        
        if selected_item:
            prompt = f"""
            The user has selected a {selected_item.category} ({selected_item.subcategory or 'general'}) from their wardrobe.
            
            Available wardrobe items: {', '.join(item_descriptions)}
            
            Suggest the best combinations with this selected item. Provide:
            1. Specific item pairings from their wardrobe
            2. Color coordination tips
            3. Style notes for each combination
            4. Occasion suitability
            
            Format your response in a friendly, conversational way.
            """
        else:
            prompt = f"""
            Based on the user's wardrobe items: {', '.join(item_descriptions)}
            
            Suggest 3-5 best outfit combinations from their existing wardrobe. For each combination:
            1. List the specific items to combine
            2. Explain why they work well together
            3. Mention color coordination
            4. Suggest suitable occasions
            
            Be specific about which items from their wardrobe to use.
            """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating outfit analysis: {e}")
            return "Sorry, I couldn't analyze your outfits right now. Please try again later."

    async def get_fashion_trends(self, query: str = "latest fashion trends 2024") -> str:
        """Get latest fashion trends using Bing Search API"""
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_api_key,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            params = {
                'q': f"{query} fashion trends",
                'textDecorations': False,
                'textFormat': 'HTML',
                'count': 5,
                'freshness': 'Month'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.bing_endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract relevant search results
                        trends_info = []
                        if 'webPages' in data and 'value' in data['webPages']:
                            for result in data['webPages']['value'][:3]:
                                trends_info.append({
                                    'title': result.get('name', ''),
                                    'snippet': result.get('snippet', ''),
                                    'url': result.get('url', '')
                                })
                        
                        # Use Gemini to summarize trends
                        trends_text = "\n".join([f"- {trend['title']}: {trend['snippet']}" for trend in trends_info])
                        
                        summary_prompt = f"""
                        Based on these fashion search results:
                        {trends_text}
                        
                        Provide a friendly summary of current fashion trends, including:
                        1. Key trending styles and colors
                        2. Popular clothing items this season
                        3. Styling tips users can apply
                        4. Color combinations that are trending
                        
                        Keep it conversational and practical.
                        """
                        
                        summary_response = self.gemini_model.generate_content(summary_prompt)
                        return summary_response.text
                    else:
                        return "I couldn't fetch the latest trends right now, but I can still help with styling advice based on your wardrobe!"
                        
        except Exception as e:
            logger.error(f"Error fetching fashion trends: {e}")
            return "I'm having trouble accessing trend information right now, but I can still help you style your existing wardrobe!"

    async def search_outfit_images(self, query: str) -> List[Dict[str, str]]:
        """Enhanced search for outfit images using Google Search API (SerpAPI)"""
        try:
            logger.info(f"Starting Google image search for: {query}")
            
            # First try: Google image search with SerpAPI
            google_images = await self._google_image_search(query)
            if len(google_images) >= 4:
                logger.info(f"Google image search successful: {len(google_images)} images")
                return google_images
            
            # Second try: Google web search for fashion sites
            fashion_images = await self._google_fashion_search(query)
            if len(fashion_images) >= 3:
                logger.info(f"Google fashion search successful: {len(fashion_images)} images")
                return fashion_images
            
            # Third try: Enhanced curated content with more variety
            curated_images = self._get_enhanced_curated_content(query)
            logger.info(f"Using enhanced curated content: {len(curated_images)} images")
            return curated_images
            
        except Exception as e:
            logger.error(f"Error in enhanced image search: {e}")
            return self._get_enhanced_curated_content(query)
    
    async def _google_image_search(self, query: str) -> List[Dict[str, str]]:
        """Google image search using SerpAPI"""
        try:
            # Enhance the query for better fashion results
            enhanced_query = f"{query} fashion outfit style clothing"
            
            params = {
                'engine': 'google_images',
                'q': enhanced_query,
                'api_key': self.google_api_key,
                'num': 20,  # Get more results to filter
                'safe': 'active',
                'hl': 'en',
                'gl': 'us'
            }
            
            logger.info(f"Searching Google Images for: {enhanced_query}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(self.google_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        images_results = data.get('images_results', [])
                        logger.info(f"Google API returned {len(images_results)} images")
                        
                        if images_results:
                            images = []
                            for img in images_results:
                                if self._is_quality_fashion_image(img, query):
                                    image_data = {
                                        'url': img.get('original', ''),
                                        'thumbnail': img.get('thumbnail', ''),
                                        'title': self._clean_image_title(img.get('title', f'{query.title()} Fashion')),
                                        'source': img.get('link', '')
                                    }
                                    images.append(image_data)
                                    
                                    if len(images) >= 8:
                                        break
                            
                            return images
                    else:
                        error_text = await response.text()
                        logger.error(f"Google API error {response.status}: {error_text}")
            
            return []
            
        except Exception as e:
            logger.error(f"Google image search failed: {e}")
            return []
    
    async def _google_fashion_search(self, query: str) -> List[Dict[str, str]]:
        """Google web search focusing on fashion sites"""
        try:
            # Search for fashion-related sites
            fashion_query = f"{query} site:zara.com OR site:hm.com OR site:asos.com OR site:uniqlo.com"
            
            params = {
                'engine': 'google',
                'q': fashion_query,
                'api_key': self.google_api_key,
                'num': 10,
                'hl': 'en',
                'gl': 'us'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(self.google_endpoint, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        organic_results = data.get('organic_results', [])
                        images = []
                        
                        for result in organic_results:
                            # Create placeholder images with shopping links
                            images.append({
                                'url': f"https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&q=80&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90oy1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                'thumbnail': f"https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=200&q=80&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                                'title': result.get('title', f'{query.title()} Fashion'),
                                'source': result.get('link', '')
                            })
                            
                            if len(images) >= 4:
                                break
                        
                        return images
            
            return []
            
        except Exception as e:
            logger.error(f"Google fashion search failed: {e}")
            return []
    
    def _is_quality_fashion_image(self, img_data: dict, query: str) -> bool:
        """Check if image is high-quality and fashion-relevant"""
        title = img_data.get('title', '').lower()
        source = img_data.get('source', '').lower()
        original_url = img_data.get('original', '')
        
        # Basic quality checks
        if not original_url:
            return False
        
        # Exclude unwanted content
        excluded_terms = [
            'anime', 'cartoon', 'drawing', 'sketch', 'illustration',
            'logo', 'text', 'diagram', 'chart', 'meme', 'advertisement'
        ]
        
        if any(term in title or term in source for term in excluded_terms):
            return False
        
        # Check relevance to query
        query_words = query.lower().split()
        relevance_score = sum(1 for word in query_words if word in title or word in source)
        
        # Prefer fashion-related content
        fashion_indicators = ['fashion', 'style', 'outfit', 'clothing', 'wear', 'trend']
        has_fashion_context = any(indicator in title or indicator in source for indicator in fashion_indicators)
        
        return relevance_score > 0 or has_fashion_context
    
    # Removed old Bing search methods - now using Google Search API
    
    # Removed old Bing search helper methods - replaced with Google Search API methods above
    
    def _clean_image_title(self, title: str) -> str:
        """Clean and improve image titles"""
        if not title:
            return "Fashion Inspiration"
        
        # Remove common unwanted text
        unwanted = ['|', 'Pinterest', 'Instagram', 'Facebook', 'Twitter']
        for term in unwanted:
            title = title.replace(term, '')
        
        # Capitalize properly
        title = title.strip().title()
        
        # Limit length
        if len(title) > 60:
            title = title[:57] + "..."
        
        return title or "Fashion Inspiration"
    
    def _remove_duplicate_images(self, images: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate images based on URL"""
        seen_urls = set()
        unique_images = []
        
        for img in images:
            url = img.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(img)
        
        return unique_images
    
    def _get_enhanced_curated_content(self, query: str) -> List[Dict[str, str]]:
        """Get enhanced curated fashion content with more variety based on query"""
        query_lower = query.lower()
        
        # Create unique images for each search to avoid repetition
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Enhanced curated content with more variety
        base_unsplash_params = "auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
        
        # Different image sets based on query content
        if any(term in query_lower for term in ['black', 'dark']):
            return [
                {
                    'url': f'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=200&q=80&{base_unsplash_params}',
                    'title': f'Elegant Black {query.title()} Style',
                    'source': f'https://www.pinterest.com/search/pins/?q={query.replace(" ", "%20")}%20black'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1583743814966-8936f37f820e?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1583743814966-8936f37f820e?w=200&q=80&{base_unsplash_params}',
                    'title': f'Classic Dark {query.title()}',
                    'source': f'https://www.zara.com/search?searchTerm={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1543076447-215ad9ba6923?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1543076447-215ad9ba6923?w=200&q=80&{base_unsplash_params}',
                    'title': f'Professional Black {query.title()}',
                    'source': f'https://www.asos.com/search/?q={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1506629905607-d9c297d3d3d3?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1506629905607-d9c297d3d3d3?w=200&q=80&{base_unsplash_params}',
                    'title': f'Sleek {query.title()} Look',
                    'source': f'https://www2.hm.com/en_us/search-results.html?q={query.replace(" ", "%20")}'
                }
            ]
        
        elif any(term in query_lower for term in ['white', 'light']):
            return [
                {
                    'url': f'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=200&q=80&{base_unsplash_params}',
                    'title': f'Classic White {query.title()}',
                    'source': f'https://www.uniqlo.com/us/en/search?q={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1598032895141-d4b9e8e6ba3b?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1598032895141-d4b9e8e6ba3b?w=200&q=80&{base_unsplash_params}',
                    'title': f'Fresh White {query.title()} Style',
                    'source': f'https://www.gap.com/browse/search.do?searchText={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?w=200&q=80&{base_unsplash_params}',
                    'title': f'Elegant Light {query.title()}',
                    'source': f'https://www.nordstrom.com/sr?keyword={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=200&q=80&{base_unsplash_params}',
                    'title': f'Minimalist {query.title()}',
                    'source': f'https://www.cos.com/en_usd/search.html?q={query.replace(" ", "%20")}'
                }
            ]
        
        elif any(term in query_lower for term in ['jean', 'denim', 'casual']):
            return [
                {
                    'url': f'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1542272604-787c3835535d?w=200&q=80&{base_unsplash_params}',
                    'title': f'Casual {query.title()} Outfit',
                    'source': f'https://www.levis.com/US/en_US/search/?q={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=200&q=80&{base_unsplash_params}',
                    'title': f'Relaxed {query.title()} Style',
                    'source': f'https://www.americaneagle.com/us/en/search?q={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1475178626620-a4d074967452?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1475178626620-a4d074967452?w=200&q=80&{base_unsplash_params}',
                    'title': f'Everyday {query.title()}',
                    'source': f'https://oldnavy.gap.com/browse/search.do?searchText={query.replace(" ", "%20")}'
                },
                {
                    'url': f'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&q=80&{base_unsplash_params}',
                    'thumbnail': f'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=200&q=80&{base_unsplash_params}',
                    'title': f'Street Style {query.title()}',
                    'source': f'https://www.urbanoutfitters.com/search?q={query.replace(" ", "%20")}'
                }
            ]
        
        # Default varied content for any other query
        return [
            {
                'url': f'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=400&q=80&{base_unsplash_params}&sig={query_hash}1',
                'thumbnail': f'https://images.unsplash.com/photo-1469334031218-e382a71b716b?w=200&q=80&{base_unsplash_params}&sig={query_hash}1',
                'title': f'Trendy {query.title()} Fashion',
                'source': f'https://www.zara.com/search?searchTerm={query.replace(" ", "%20")}'
            },
            {
                'url': f'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=400&q=80&{base_unsplash_params}&sig={query_hash}2',
                'thumbnail': f'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=200&q=80&{base_unsplash_params}&sig={query_hash}2',
                'title': f'Stylish {query.title()} Inspiration',
                'source': f'https://www.asos.com/search/?q={query.replace(" ", "%20")}'
            },
            {
                'url': f'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&q=80&{base_unsplash_params}&sig={query_hash}3',
                'thumbnail': f'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=200&q=80&{base_unsplash_params}&sig={query_hash}3',
                'title': f'Modern {query.title()} Look',
                'source': f'https://www2.hm.com/en_us/search-results.html?q={query.replace(" ", "%20")}'
            },
            {
                'url': f'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&q=80&{base_unsplash_params}&sig={query_hash}4',
                'thumbnail': f'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=200&q=80&{base_unsplash_params}&sig={query_hash}4',
                'title': f'Chic {query.title()} Style',
                'source': f'https://www.uniqlo.com/us/en/search?q={query.replace(" ", "%20")}'
            }
        ]
    
    async def _search_fashion_api(self, query: str) -> List[Dict[str, str]]:
        """Try fashion-specific search approach with updated API"""
        try:
            # Use very specific fashion search terms
            fashion_query = self._enhance_fashion_query(query)
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_api_key,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            params = {
                'q': fashion_query,
                'imageType': 'Photo',
                'size': 'Medium',
                'aspect': 'Portrait',  # Better for clothing
                'color': 'ColorOnly',
                'count': 15,  # Get more results to filter
                'safeSearch': 'Moderate',
                'market': 'en-US',
                'setLang': 'en',
                'freshness': 'Month'
            }
            
            endpoint = "https://api.bing.microsoft.com/v7.0/images/search"
            
            logger.info(f"Searching Bing Images with updated API for: {fashion_query}")
            logger.info(f"Using API key: {self.bing_api_key[:15]}...")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(endpoint, headers=headers, params=params) as response:
                    logger.info(f"Bing API response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Bing API returned {len(data.get('value', []))} total images")
                        
                        if 'value' in data and len(data['value']) > 0:
                            images = []
                            for img in data['value']:
                                title = img.get('name', '').lower()
                                content_url = img.get('contentUrl', '')
                                host_url = img.get('hostPageUrl', '').lower()
                                
                                # Flexible filtering for any fashion item
                                is_relevant = (
                                    self._is_fashion_relevant(title, query) and 
                                    content_url and
                                    # Exclude non-fashion sites
                                    not any(skip in host_url for skip in ['pinterest.com', 'etsy.com', 'ebay.com', 'aliexpress.com', 'alibaba.com']) and
                                    # Include fashion sites or general fashion content
                                    (any(fashion_site in host_url for fashion_site in ['zara.com', 'hm.com', 'asos.com', 'uniqlo.com', 'nordstrom.com', 'macys.com', 'gap.com', 'forever21.com']) or
                                     self._has_fashion_content(title, query))
                                )
                                
                                if is_relevant:
                                    images.append({
                                        'url': content_url,
                                        'thumbnail': img.get('thumbnailUrl', content_url),
                                        'title': img.get('name', f'{query.title()} Fashion'),
                                        'source': img.get('hostPageUrl', '')
                                    })
                                    
                                    logger.info(f"Added relevant image: {title[:50]}...")
                                    
                                    if len(images) >= 6:
                                        break
                            
                            logger.info(f"Filtered to {len(images)} relevant fashion images")
                            return images
                    elif response.status == 401:
                        logger.error("Bing API authentication failed - API key invalid")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                    elif response.status == 403:
                        logger.error("Bing API access forbidden")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Bing API error {response.status}: {error_text}")
            
            return []
            
        except Exception as e:
            logger.error(f"Fashion API search failed: {e}")
            return []
    
    async def _search_bing_enhanced(self, query: str) -> List[Dict[str, str]]:
        """Enhanced Bing search with better parameters"""
        try:
            # Try web search API instead of image search
            headers = {
                'Ocp-Apim-Subscription-Key': self.bing_api_key,
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Use web search to get shopping results
            shopping_query = f"{query} shop buy online fashion clothing store"
            
            params = {
                'q': shopping_query,
                'count': 8,
                'responseFilter': 'Webpages',
                'safeSearch': 'Moderate',
                'market': 'en-US'
            }
            
            endpoint = "https://api.bing.microsoft.com/v7.0/search"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(endpoint, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'webPages' in data and 'value' in data['webPages']:
                            # Extract fashion shopping sites
                            images = []
                            for page in data['webPages']['value']:
                                url = page.get('url', '')
                                name = page.get('name', '')
                                
                                # Focus on known fashion retailers
                                if any(site in url.lower() for site in ['zara.com', 'hm.com', 'asos.com', 'uniqlo.com', 'amazon.com']):
                                    images.append({
                                        'url': f"https://via.placeholder.com/400x600/000000/FFFFFF?text={query.replace(' ', '+')}",
                                        'thumbnail': f"https://via.placeholder.com/200x300/000000/FFFFFF?text={query.replace(' ', '+')}",
                                        'title': f"{query.title()} - {name[:50]}",
                                        'source': url
                                    })
                                    
                                    if len(images) >= 4:
                                        break
                            
                            return images
            
            return []
            
        except Exception as e:
            logger.error(f"Enhanced Bing search failed: {e}")
            return []
    
    def _enhance_fashion_query(self, query: str) -> str:
        """Enhance query with fashion-specific terms for any clothing item"""
        query_lower = query.lower()
        
        # Add fashion context while preserving the original search intent
        fashion_terms = "fashion clothing apparel style outfit wear"
        
        # Keep the original query and add fashion context
        enhanced_query = f"{query} {fashion_terms}"
        
        # Add specific category terms if detected
        if any(term in query_lower for term in ['pant', 'trouser']):
            enhanced_query += " pants trousers slacks"
        elif any(term in query_lower for term in ['shirt', 'blouse']):
            enhanced_query += " shirt blouse top"
        elif any(term in query_lower for term in ['dress']):
            enhanced_query += " dress gown formal"
        elif any(term in query_lower for term in ['jean']):
            enhanced_query += " jeans denim"
        elif any(term in query_lower for term in ['shoe', 'boot', 'sneaker']):
            enhanced_query += " shoes footwear boots sneakers"
        elif any(term in query_lower for term in ['jacket', 'coat', 'blazer']):
            enhanced_query += " jacket coat blazer outerwear"
        
        return enhanced_query
    
    def _is_fashion_relevant(self, title: str, query: str) -> bool:
        """Check if image title is relevant to any fashion query"""
        fashion_keywords = [
            'fashion', 'clothing', 'style', 'outfit', 'wear', 'apparel',
            'dress', 'shirt', 'pants', 'trouser', 'jeans', 'blouse', 'top',
            'jacket', 'coat', 'blazer', 'sweater', 'hoodie', 'cardigan',
            'shoes', 'boots', 'sneakers', 'sandals', 'heels', 'footwear',
            'skirt', 'shorts', 'swimwear', 'lingerie', 'accessories',
            'suit', 'formal', 'casual', 'business', 'evening', 'party'
        ]
        
        query_words = query.lower().split()
        title_lower = title.lower()
        
        # Check if title contains fashion keywords
        has_fashion = any(keyword in title_lower for keyword in fashion_keywords)
        
        # Check if title contains any words from the user's query
        has_query_match = any(word in title_lower for word in query_words if len(word) > 2)
        
        # Also check for color and material terms
        color_terms = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'gray', 'grey', 'navy', 'beige']
        material_terms = ['cotton', 'silk', 'wool', 'leather', 'denim', 'polyester', 'linen']
        
        has_descriptive = any(term in title_lower for term in color_terms + material_terms)
        
        # Image is relevant if it has fashion context AND (matches query OR has descriptive terms)
        return has_fashion and (has_query_match or has_descriptive)
    
    def _has_fashion_content(self, title: str, query: str) -> bool:
        """Check if the content appears to be fashion-related based on title"""
        title_lower = title.lower()
        
        # Check for any clothing-related terms
        clothing_indicators = [
            'wear', 'style', 'look', 'outfit', 'fashion', 'trend', 'chic', 'elegant',
            'casual', 'formal', 'vintage', 'modern', 'classic', 'designer',
            'collection', 'wardrobe', 'attire', 'garment', 'textile'
        ]
        
        return any(indicator in title_lower for indicator in clothing_indicators)
    
    def _get_fashion_specific_images(self, query: str) -> List[Dict[str, str]]:
        """Get high-quality fashion-specific images based on query"""
        query_lower = query.lower()
        
        # High-quality curated fashion images
        if 'black' in query_lower and ('pant' in query_lower or 'trouser' in query_lower):
            return [
                {
                    'url': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400&q=80&auto=format&fit=crop',
                    'thumbnail': 'https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=200&q=80&auto=format&fit=crop',
                    'title': 'Classic Black Dress Pants',
                    'source': 'https://www.zara.com/search?searchTerm=black%20pants'
                },
                {
                    'url': 'https://images.unsplash.com/photo-1543076447-215ad9ba6923?w=400&q=80&auto=format&fit=crop',
                    'thumbnail': 'https://images.unsplash.com/photo-1543076447-215ad9ba6923?w=200&q=80&auto=format&fit=crop',
                    'title': 'Black Formal Trousers',
                    'source': 'https://www.asos.com/search/?q=black%20pants'
                },
                {
                    'url': 'https://images.unsplash.com/photo-1506629905607-d9c297d3d3d3?w=400&q=80&auto=format&fit=crop',
                    'thumbnail': 'https://images.unsplash.com/photo-1506629905607-d9c297d3d3d3?w=200&q=80&auto=format&fit=crop',
                    'title': 'Black Slim Fit Pants',
                    'source': 'https://www2.hm.com/en_us/search-results.html?q=black%20pants'
                },
                {
                    'url': 'https://images.unsplash.com/photo-1475178626620-a4d074967452?w=400&q=80&auto=format&fit=crop',
                    'thumbnail': 'https://images.unsplash.com/photo-1475178626620-a4d074967452?w=200&q=80&auto=format&fit=crop',
                    'title': 'Professional Black Pants',
                    'source': 'https://www.amazon.com/s?k=black+dress+pants'
                }
            ]
        elif 'white' in query_lower and 'shirt' in query_lower:
            return [
                {
                    'url': 'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=400&q=80',
                    'thumbnail': 'https://images.unsplash.com/photo-1562157873-818bc0726f68?w=200&q=80',
                    'title': 'Classic White Dress Shirt',
                    'source': 'https://www.zara.com/search?searchTerm=white%20shirt'
                },
                {
                    'url': 'https://images.unsplash.com/photo-1598032895141-d4b9e8e6ba3b?w=400&q=80',
                    'thumbnail': 'https://images.unsplash.com/photo-1598032895141-d4b9e8e6ba3b?w=200&q=80',
                    'title': 'White Button-up Shirt',
                    'source': 'https://www.asos.com/search/?q=white%20shirt'
                }
            ]
        else:
            # Generic fashion images with shopping links
            return [
                {
                    'url': 'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=400&q=80',
                    'thumbnail': 'https://images.unsplash.com/photo-1441986300917-64674bd600d8?w=200&q=80',
                    'title': f'Fashion {query.title()}',
                    'source': f'https://www.zara.com/search?searchTerm={query.replace(" ", "%20")}'
                },
                {
                    'url': 'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=400&q=80',
                    'thumbnail': 'https://images.unsplash.com/photo-1558769132-cb1aea458c5e?w=200&q=80',
                    'title': f'Stylish {query.title()}',
                    'source': f'https://www.asos.com/search/?q={query.replace(" ", "%20")}'
                }
            ]

    async def find_missing_wardrobe_items(self, user: User, db: Session) -> str:
        """Analyze user's wardrobe and suggest missing essential items"""
        wardrobe_context = await self.get_user_wardrobe_context(user, db)
        
        # Essential wardrobe checklist
        essentials = {
            'tops': ['t-shirt', 'dress shirt', 'blouse', 'sweater'],
            'bottoms': ['jeans', 'formal trousers', 'shorts'],
            'shoes': ['sneakers', 'dress shoes', 'boots'],
            'outerwear': ['jacket', 'blazer', 'coat'],
            'accessories': ['belt', 'watch']
        }
        
        prompt = f"""
        Analyze this user's wardrobe:
        {wardrobe_context}
        
        Essential wardrobe items: {json.dumps(essentials)}
        
        Based on their current wardrobe:
        1. Identify what essential items they're missing
        2. Suggest specific pieces that would complement their existing items
        3. Prioritize recommendations (most important first)
        4. Consider their existing style and color palette
        5. Suggest versatile pieces that work with multiple outfits
        
        Provide practical, budget-conscious recommendations.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing missing items: {e}")
            return "I couldn't analyze your wardrobe gaps right now. Please try again later."

    async def get_style_advice(self, user: User, db: Session, question: str) -> str:
        """Provide personalized style advice based on user's wardrobe and question"""
        wardrobe_context = await self.get_user_wardrobe_context(user, db)
        
        # User profile context
        user_context = f"""User Profile:
        • Gender: {user.gender.value if user.gender else 'not specified'}
        • Age: {user.age}
        • Height: {user.height}cm
        
        {wardrobe_context}
        """
        
        prompt = f"""{self.fashion_context}
        
        {user_context}
        
        User's question: "{question}"
        
        Provide personalized fashion advice considering:
        1. Their existing wardrobe items (be specific about what they own)
        2. Their personal profile
        3. Current fashion trends
        4. Practical styling tips
        5. Color coordination
        6. Occasion-appropriate suggestions
        
        Keep your response concise, practical, and easy to read. Use simple formatting without asterisks.
        When referencing their wardrobe, be specific about the actual items they have.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error providing style advice: {e}")
            return "I'm having trouble accessing my styling knowledge right now. Please try asking again!"

    async def process_message(self, user: User, message: str, db: Session, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced message processing with wardrobe context, web search integration, and chat history"""
        message_lower = message.lower()
        
        # Get recent chat history for context
        recent_history = self.get_recent_chat_context(user, db, limit=6)  # Last 6 messages for context
        
        # Save user message to database
        chat_message = ChatMessage(
            user_id=user.id,
            message=message,
            is_from_user=True,
            message_type="general"
        )
        db.add(chat_message)
        db.commit()
        
        response_data = {
            "response": "",
            "type": "text",
            "images": [],
            "suggestions": []
        }
        
        try:
            # Check if user has selected an item from wardrobe
            selected_item = None
            if context and context.get('selected_item_id'):
                selected_item = db.query(Outfit).filter(
                    Outfit.id == context['selected_item_id'],
                    Outfit.user_id == user.id
                ).first()
            
            # Handle wardrobe item-specific queries
            if selected_item:
                response_data = await self.handle_selected_item_query(user, db, message, selected_item)
            
            # PRIORITY: Handle explicit web search requests FIRST
            elif any(keyword in message_lower for keyword in [
                'search me', 'find me', 'show me', 'from web', 'web search', 
                'search web', 'find from web', 'show me from web',
                'search for', 'look for', 'get me', 'buy', 'purchase'
            ]):
                response_data = await self.handle_web_search_query(message)
            
            # Enhanced wardrobe query detection (only if NOT a web search request)
            elif any(keyword in message_lower for keyword in ['how many', 'do i have', 'have i got', 'count', 'what shirts', 'what pants', 'what dresses', 'what shoes', 'which color', 'most common']):
                # Wardrobe counting and analysis queries
                if any(count_word in message_lower for count_word in ['how many', 'count', 'total']):
                    # Check if asking about specific category
                    specific_categories = ['shirt', 'shirts', 'pants', 'shoes', 'dress', 'dresses', 'jacket', 'jackets', 'hat', 'hats']
                    found_category = None
                    for cat in specific_categories:
                        if cat in message_lower:
                            found_category = cat.rstrip('s')  # Remove plural 's'
                            break
                    
                    if found_category:
                        # Answer about specific category only
                        search_result = await self.search_wardrobe_items(message, user.id)
                        response_data["response"] = search_result
                        response_data["type"] = "specific_count"
                    else:
                        # Show full wardrobe summary
                        wardrobe_context = await self.get_user_wardrobe_context(user, db)
                        response_data["response"] = wardrobe_context
                        response_data["type"] = "wardrobe_count"
                else:
                    # Extract search terms for specific items
                    search_terms = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'grey', 'navy',
                                  'shirt', 'pants', 'jeans', 'dress', 'jacket', 'shoes', 'boots', 'sneakers', 'blouse', 'sweater', 'skirt']
                    found_terms = [term for term in search_terms if term in message_lower]
                    
                    if found_terms:
                        search_query = ' '.join(found_terms)
                        search_result = await self.search_wardrobe_items(message, user.id)
                        response_data["response"] = search_result
                        response_data["type"] = "wardrobe_search"
                    else:
                        wardrobe_context = await self.get_user_wardrobe_context(user, db)
                        response_data["response"] = f"Here's what I can see in your wardrobe:\n\n{wardrobe_context}\n\nWhat specific item are you looking for?"
                        response_data["type"] = "wardrobe_overview"
                        
            elif any(keyword in message_lower for keyword in ['trend', 'trending', 'fashion trend', 'what\'s popular']):
                response_data["response"] = await self.get_fashion_trends(message)
                response_data["type"] = "trends"
                
            elif any(keyword in message_lower for keyword in ['combination', 'combo', 'pair with', 'match with', 'goes with']):
                selected_item_id = context.get('selected_item_id') if context else None
                response_data["response"] = await self.analyze_outfit_combination(user, db, selected_item_id)
                response_data["type"] = "combination"
                
            elif any(keyword in message_lower for keyword in ['missing', 'need', 'should buy', 'wardrobe gap', 'essential']):
                response_data["response"] = await self.find_missing_wardrobe_items(user, db)
                response_data["type"] = "wardrobe_analysis"
                
            else:
                # General style advice with wardrobe context and chat history
                response_data["response"] = await self.get_style_advice_with_history(user, db, message, recent_history)
                response_data["type"] = "advice"
            
            # Clean up response - remove asterisks and improve formatting
            response_data["response"] = self.clean_response_formatting(response_data["response"])
            
            # Save AI response to database
            ai_response = ChatMessage(
                user_id=user.id,
                message=message,
                response=response_data["response"],
                is_from_user=False,
                message_type=response_data["type"],
                context_data=json.dumps(context) if context else None
            )
            db.add(ai_response)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response_data["response"] = "I'm having some technical difficulties right now. Please try again in a moment!"
            response_data["type"] = "error"
        
        return response_data

    async def handle_selected_item_query(self, user: User, db: Session, message: str, selected_item) -> Dict[str, Any]:
        """Handle queries about selected wardrobe items with enhanced visual matching"""
        response_data = {
            "response": "",
            "type": "selected_item",
            "images": [],
            "suggestions": [],
            "wardrobe_matches": []  # New field for actual wardrobe items
        }
        
        try:
            message_lower = message.lower()
            
            # Get user's wardrobe for matching suggestions
            all_items = db.query(Outfit).filter(Outfit.user_id == user.id).all()
            
            # Check if looking for matching items
            if any(keyword in message_lower for keyword in ['match', 'pair', 'goes with', 'combine', 'wear with', 'what to wear', 'outfit', 'style']):
                # Enhanced matching algorithm with better logic
                matching_items = self._find_smart_matches(selected_item, all_items)
                
                if matching_items:
                    response_data["wardrobe_matches"] = matching_items
                    response_data["response"] = f"🎯 Perfect! Here are items from YOUR wardrobe that would look amazing with your {selected_item.subcategory or selected_item.category}:\n\n"
                    
                    # Group suggestions by category for better presentation
                    category_groups = {}
                    for item in matching_items:
                        cat = item["category"]
                        if cat not in category_groups:
                            category_groups[cat] = []
                        category_groups[cat].append(item["subcategory"] or "general")
                    
                    for cat, items in category_groups.items():
                        response_data["response"] += f"👕 {cat.title()}: {', '.join(set(items))}\n"
                    
                    # Add AI-powered styling advice
                    styling_advice = await self._get_ai_styling_advice(selected_item, matching_items)
                    response_data["response"] += f"\n✨ {styling_advice}"
                    
                    response_data["response"] += "\n\n🖼️ Click on any item below to see how they look together!"
                    
                    # If limited wardrobe matches, search web for inspiration
                    if len(matching_items) < 3:
                        response_data["response"] += "\n\n🌐 Want more styling ideas? Here are some inspirations from top fashion sources:"
                        web_query = f"how to style {selected_item.subcategory or selected_item.category} outfit ideas fashion"
                        web_images = await self.search_outfit_images(web_query)
                        if web_images:
                            response_data["images"] = web_images[:4]
                            response_data["response"] += "\n\n📸 Swipe through these professional styling inspirations!"
                else:
                    # No wardrobe matches - provide comprehensive suggestions
                    response_data["response"] = f"🔍 You don't have matching items in your wardrobe for your {selected_item.subcategory or selected_item.category} yet.\n\n"
                    
                    # Suggest what items would work well
                    suggestions = self._get_missing_pieces_suggestions(selected_item)
                    response_data["response"] += f"💡 Here's what would pair perfectly:\n{suggestions}\n\n"
                    
                    # Show web inspirations
                    response_data["response"] += "🌟 Meanwhile, here are some professional styling ideas:"
                    web_query = f"how to wear {selected_item.subcategory or selected_item.category} styling outfit ideas"
                    web_images = await self.search_outfit_images(web_query)
                    if web_images:
                        response_data["images"] = web_images
                        response_data["response"] += "\n\n✨ Get inspired by these professional looks!"
            
            # General questions about the selected item
            else:
                wardrobe_context = await self.get_user_wardrobe_context(user, db)
                
                prompt = f"""{self.fashion_context}
                
                The user has selected this item from their wardrobe:
                - Category: {selected_item.category}
                - Subcategory: {selected_item.subcategory or 'general'}
                - Colors: {selected_item.primary_color or 'unknown'}
                
                User's complete wardrobe: {wardrobe_context}
                
                User's question: "{message}"
                
                Provide specific advice about this selected item considering:
                1. How to style it with other items from their wardrobe
                2. Color coordination tips
                3. Occasion-appropriate styling
                4. Care and maintenance tips if relevant
                5. Current trend relevance
                
                Be specific about their actual wardrobe items and keep the response engaging and practical.
                """
                
                try:
                    ai_response = self.gemini_model.generate_content(prompt)
                    response_data["response"] = ai_response.text
                except Exception as e:
                    logger.error(f"Error getting AI response for selected item: {e}")
                    response_data["response"] = f"I can see you've selected your {selected_item.subcategory or selected_item.category}! Ask me about styling it, what to pair it with, or any specific questions about this item."
            
        except Exception as e:
            logger.error(f"Error handling selected item query: {e}")
            response_data["response"] = "I had trouble processing your question about the selected item. Please try again!"
        
        return response_data

    def _find_smart_matches(self, selected_item, all_items) -> List[Dict[str, Any]]:
        """Enhanced algorithm to find clothing matches with better logic"""
        matching_items = []
        selected_category = selected_item.category.lower()
        selected_subcategory = (selected_item.subcategory or "").lower()
        selected_color = (selected_item.primary_color or "").lower()
        
        # Define comprehensive matching rules
        matching_rules = {
            # Tops (shirts, blouses, t-shirts, etc.)
            'top': ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'trousers', 'shoes', 'jacket', 'blazer'],
            'shirt': ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'trousers', 'shoes', 'jacket', 'blazer'],
            't-shirt': ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'trousers', 'shoes', 'jacket'],
            'blouse': ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'trousers', 'shoes', 'jacket', 'blazer'],
            'tank': ['bottom', 'pants', 'jeans', 'skirt', 'shorts', 'trousers', 'shoes', 'jacket'],
            
            # Bottoms
            'bottom': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket', 'blazer'],
            'pants': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket', 'blazer'],
            'jeans': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket', 'blazer'],
            'shorts': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket'],
            'skirt': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket', 'blazer'],
            'trousers': ['top', 'shirt', 'blouse', 't-shirt', 'tank', 'shoes', 'jacket', 'blazer'],
            
            # Dresses
            'dress': ['shoes', 'jacket', 'blazer', 'cardigan', 'coat'],
            
            # Outerwear
            'jacket': ['top', 'shirt', 'blouse', 't-shirt', 'bottom', 'pants', 'jeans', 'dress', 'shoes'],
            'blazer': ['top', 'shirt', 'blouse', 'bottom', 'pants', 'jeans', 'dress', 'shoes'],
            'cardigan': ['top', 'shirt', 'blouse', 'bottom', 'pants', 'jeans', 'dress', 'shoes'],
            'coat': ['dress', 'shoes'],
            
            # Shoes (match with everything)
            'shoes': ['top', 'shirt', 'bottom', 'pants', 'jeans', 'dress', 'skirt', 'shorts'],
            'boots': ['top', 'shirt', 'bottom', 'pants', 'jeans', 'skirt'],
            'sneakers': ['top', 'shirt', 'bottom', 'pants', 'jeans', 'shorts'],
            'sandals': ['top', 'shirt', 'bottom', 'shorts', 'skirt', 'dress']
        }
        
        # Find matching categories
        compatible_categories = matching_rules.get(selected_category, [])
        if selected_subcategory in matching_rules:
            compatible_categories.extend(matching_rules[selected_subcategory])
        
        # Score and filter matches
        for item in all_items:
            if item.id == selected_item.id:
                continue
                
            item_category = item.category.lower()
            item_subcategory = (item.subcategory or "").lower()
            item_color = (item.primary_color or "").lower()
            
            # Check if categories are compatible
            is_compatible = (
                item_category in compatible_categories or
                item_subcategory in compatible_categories or
                any(cat in item_category for cat in compatible_categories) or
                any(cat in item_subcategory for cat in compatible_categories)
            )
            
            if is_compatible:
                # Calculate match score
                score = self._calculate_match_score(selected_item, item)
                
                matching_items.append({
                    "id": item.id,
                    "category": item.category,
                    "subcategory": item.subcategory,
                    "image_path": item.image_path,
                    "primary_color": item.primary_color,
                    "score": score
                })
        
        # Sort by score and return top matches
        matching_items.sort(key=lambda x: x['score'], reverse=True)
        return matching_items[:8]  # Return top 8 matches

    def _calculate_match_score(self, selected_item, match_item) -> float:
        """Calculate compatibility score between two items"""
        score = 1.0
        
        # Color compatibility bonus
        if selected_item.primary_color and match_item.primary_color:
            selected_color = selected_item.primary_color.lower()
            match_color = match_item.primary_color.lower()
            
            # Define color compatibility
            color_compatibility = {
                'black': ['white', 'gray', 'navy', 'red', 'blue', 'green'],
                'white': ['black', 'navy', 'blue', 'red', 'gray'],
                'gray': ['black', 'white', 'navy', 'blue', 'pink'],
                'navy': ['white', 'gray', 'beige', 'red'],
                'blue': ['white', 'gray', 'beige', 'navy'],
                'red': ['black', 'white', 'navy', 'gray'],
                'beige': ['navy', 'blue', 'brown', 'white'],
                'brown': ['beige', 'white', 'cream']
            }
            
            if match_color in color_compatibility.get(selected_color, []):
                score += 0.5
            elif selected_color == match_color:
                score += 0.2  # Same color is okay but not always best
        
        # Category combination bonus
        selected_cat = selected_item.category.lower()
        match_cat = match_item.category.lower()
        
        # Perfect combinations get higher scores
        perfect_combos = [
            ('shirt', 'pants'), ('blouse', 'skirt'), ('t-shirt', 'jeans'),
            ('dress', 'jacket'), ('top', 'bottom'), ('jacket', 'dress')
        ]
        
        for combo in perfect_combos:
            if (selected_cat in combo[0] and match_cat in combo[1]) or \
               (selected_cat in combo[1] and match_cat in combo[0]):
                score += 0.3
                break
        
        return score

    async def _get_ai_styling_advice(self, selected_item, matching_items) -> str:
        """Get AI-powered styling advice for the combination"""
        try:
            items_desc = [f"{item['category']}/{item['subcategory']}" for item in matching_items[:3]]
            
            prompt = f"""
            Give a brief (2-3 sentences) styling tip for wearing a {selected_item.category}/{selected_item.subcategory} 
            with these items: {', '.join(items_desc)}.
            
            Focus on:
            - Color coordination
            - Style harmony
            - Occasion suitability
            
            Keep it friendly and practical.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error getting styling advice: {e}")
            return "Mix and match these pieces for a stylish, coordinated look!"

    def _get_missing_pieces_suggestions(self, selected_item) -> str:
        """Suggest what pieces would complete the outfit"""
        category = selected_item.category.lower()
        
        suggestions = {
            'shirt': "• Dark jeans or chinos\n• Dress shoes or loafers\n• Optional blazer for formal occasions",
            'blouse': "• Tailored trousers or pencil skirt\n• Heels or flats\n• Statement jewelry",
            't-shirt': "• Jeans or casual shorts\n• Sneakers or casual shoes\n• Denim jacket or cardigan",
            'dress': "• Cardigan or blazer\n• Appropriate shoes (heels/flats)\n• Accessories to complete the look",
            'pants': "• Matching tops (shirts, blouses)\n• Shoes that complement the style\n• Belt if needed",
            'jeans': "• Casual tops (t-shirts, shirts)\n• Sneakers or boots\n• Light jacket or sweater",
        }
        
        return suggestions.get(category, "• Complementary tops or bottoms\n• Appropriate footwear\n• Accessories to enhance the style")

    async def handle_web_search_query(self, message: str) -> Dict[str, Any]:
        """Handle web search requests for fashion items"""
        response_data = {
            "response": "",
            "type": "web_search",
            "images": [],
            "suggestions": []
        }
        
        try:
            # Extract search terms from message more intelligently
            message_lower = message.lower()
            logger.info(f"Processing web search query: {message}")
            
            # Remove common web search phrases to extract the actual search term
            search_query = message_lower
            
            # Remove search command phrases
            remove_phrases = [
                'can you search me', 'search me', 'find me', 'show me', 'get me',
                'search for', 'look for', 'find from web', 'search from web',
                'from web', 'web search', 'search web', 'show me from web',
                'can you find', 'can you show', 'i want', 'i need'
            ]
            
            for phrase in remove_phrases:
                search_query = search_query.replace(phrase, '')
            
            # Clean up the search query
            search_query = search_query.strip()
            logger.info(f"Extracted search query: '{search_query}'")
            
            # If no specific item mentioned after cleaning, use a default
            if not search_query or len(search_query) < 2:
                search_query = "black pants"
                logger.info(f"Using default search query: '{search_query}'")
            
            # Search for images (multiple fallback methods)
            web_images = await self.search_outfit_images(search_query)
            
            # Always provide images and direct shopping guidance
            response_data["images"] = web_images
            
            # Create clickable shopping links (return as structured data for frontend)
            search_encoded = search_query.replace(' ', '%20')
            shopping_links = [
                {"name": "Zara", "url": f"https://www.zara.com/search?searchTerm={search_encoded}"},
                {"name": "ASOS", "url": f"https://www.asos.com/search/?q={search_encoded}"},
                {"name": "H&M", "url": f"https://www2.hm.com/en_us/search-results.html?q={search_encoded}"},
                {"name": "Amazon", "url": f"https://www.amazon.com/s?k={search_query.replace(' ', '+')}"},
                {"name": "Uniqlo", "url": f"https://www.uniqlo.com/us/en/search?q={search_encoded}"}
            ]
            response_data["shopping_links"] = shopping_links
            
            # Create a simple text response with direct guidance
            response_data["response"] = f"🛍️ **Shopping for {search_query.title()}:**\n\n"
            response_data["response"] += "Here are the best places to find exactly what you're looking for:\n\n"
            response_data["response"] += "🌐 **Direct Shopping Links:**\n"
            for link in shopping_links:
                response_data["response"] += f"• {link['name']}: Click the link button below\n"
            response_data["response"] += "\n📸 **Fashion Inspiration:**\n"
            response_data["response"] += "Browse the images below for styling ideas (click any image to visit the source)\n\n"
            
            # Add practical shopping and styling advice
            styling_prompt = f"""
            A user is shopping for "{search_query}" online.
            
            Provide concise, practical advice in this format:
            
            **What to Look For:**
            - Key quality indicators and features
            - Sizing and fit tips
            
            **Styling Ideas:**
            - How to wear and style these items
            - What to pair them with
            
            **Shopping Tips:**
            - Price ranges to expect
            - Best times to shop/sales
            
            Keep it under 200 words, use bullet points, be specific and helpful.
            """
            
            try:
                ai_response = self.gemini_model.generate_content(styling_prompt)
                response_data["response"] += ai_response.text
            except Exception as e:
                logger.error(f"Error getting AI response for web search: {e}")
                response_data["response"] += f"✨ **Quick Shopping Guide:**\n\n"
                response_data["response"] += "**What to Look For:**\n"
                response_data["response"] += "• Check fabric composition and care instructions\n"
                response_data["response"] += "• Read size guides and customer reviews\n"
                response_data["response"] += "• Look for return/exchange policies\n\n"
                response_data["response"] += "**Styling Tips:**\n"
                response_data["response"] += "• Choose versatile pieces that match your wardrobe\n"
                response_data["response"] += "• Consider the occasions you'll wear them\n"
                response_data["response"] += "• Think about color coordination with existing items\n\n"
                response_data["response"] += "**Shopping Tips:**\n"
                if 'black' in search_query and 'pant' in search_query:
                    response_data["response"] += "• Black pants: $25-80 for good quality\n"
                    response_data["response"] += "• Try searching: 'black dress pants', 'black trousers', 'black work pants'\n"
                else:
                    response_data["response"] += f"• Compare prices across multiple stores\n"
                    response_data["response"] += f"• Check for sales and discount codes\n"
            
            response_data["response"] += "\n\n📱 **Pro Tip:** Use the shopping apps of these stores for exclusive deals and notifications!"
            
        except Exception as e:
            logger.error(f"Error handling web search query: {e}")
            response_data["response"] = "I had trouble searching the web right now. Please try again or ask me about styling your existing wardrobe!"
        
        return response_data

    def format_response_with_points(self, response: str) -> str:
        """Format response with bullet points where necessary and improve readability"""
        if not response:
            return response
        
        # Clean the response first
        response = self.clean_response_formatting(response)
        
        # Check if response would benefit from bullet points
        # Look for numbered lists, multiple suggestions, or long paragraphs
        if any(pattern in response.lower() for pattern in [
            '1.', '2.', '3.',  # Numbered lists
            'first', 'second', 'third',  # Sequential indicators
            'tips:', 'suggestions:', 'options:', 'ideas:',  # List indicators
            'here are', 'you can', 'consider'
        ]):
            # Split into sentences and format appropriately
            sentences = response.split('. ')
            formatted_parts = []
            current_section = []
            
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if this sentence starts a new point/tip
                if any(starter in sentence.lower()[:20] for starter in [
                    'tip:', 'suggestion:', 'option:', 'idea:',
                    '1.', '2.', '3.', '4.', '5.',
                    'first', 'second', 'third', 'fourth', 'fifth',
                    'you can', 'try', 'consider'
                ]):
                    # Save previous section
                    if current_section:
                        formatted_parts.append('. '.join(current_section) + '.')
                        current_section = []
                    
                    # Start new point with bullet
                    if not sentence.startswith('•'):
                        sentence = f'• {sentence}'
                
                current_section.append(sentence)
            
            # Add remaining section
            if current_section:
                formatted_parts.append('. '.join(current_section) + ('.' if not current_section[-1].endswith('.') else ''))
            
            # Join with proper spacing
            response = '\n\n'.join(formatted_parts)
        
        # Final cleanup - ensure bullet points are properly spaced
        lines = response.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('•') and formatted_lines and not formatted_lines[-1].strip() == '':
                formatted_lines.append('')  # Add spacing before bullet points
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines).strip()

    def clean_response_formatting(self, text: str) -> str:
        """Clean response text to remove asterisks and improve readability"""
        if not text:
            return text
            
        # Remove asterisks used for emphasis
        text = text.replace('**', '').replace('*', '')
        
        # Clean up markdown-style formatting
        text = text.replace('###', '').replace('##', '').replace('#', '')
        
        # Improve bullet points
        text = text.replace('- ', '• ')
        
        # Clean up excessive whitespace and long paragraphs
        import re
        
        # Split very long paragraphs into shorter ones
        sentences = text.split('. ')
        formatted_sentences = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            # Break paragraph after 2-3 sentences or if it gets too long
            if (len(current_paragraph) >= 2 and len(' '.join(current_paragraph)) > 150) or len(current_paragraph) >= 3:
                formatted_sentences.append('. '.join(current_paragraph) + ('.' if not current_paragraph[-1].endswith('.') else ''))
                current_paragraph = []
        
        if current_paragraph:
            formatted_sentences.append('. '.join(current_paragraph) + ('.' if not current_paragraph[-1].endswith('.') else ''))
        
        text = '\n\n'.join(formatted_sentences)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Limit total response length to be more concise
        if len(text) > 600:
            # For jacket or styling advice, cut it even shorter
            if any(word in text.lower() for word in ['jacket', 'styling', 'pair', 'outfit']):
                text = text[:300] + "\n\n💡 Need styling tips? Just ask!"
            else:
                text = text[:600] + "\n\n💡 Ask me for more specific details if you'd like!"
        
        return text.strip()

    def get_recent_chat_context(self, user: User, db: Session, limit: int = 6) -> str:
        """Get recent chat history for context maintenance"""
        try:
            # Get recent messages (both user and AI)
            recent_messages = db.query(ChatMessage).filter(
                ChatMessage.user_id == user.id
            ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
            
            if not recent_messages:
                return "No previous conversation context."
            
            # Build context string
            context_lines = []
            for msg in reversed(recent_messages):  # Reverse to get chronological order
                if msg.is_from_user:
                    context_lines.append(f"User: {msg.message}")
                elif msg.response:
                    # Truncate long AI responses for context
                    response_text = msg.response[:150] + "..." if len(msg.response) > 150 else msg.response
                    context_lines.append(f"Assistant: {response_text}")
            
            return "\n".join(context_lines)
            
        except Exception as e:
            logger.error(f"Error getting chat context: {e}")
            return "No previous conversation context."
    
    async def get_style_advice_with_history(self, user: User, db: Session, question: str, chat_history: str) -> str:
        """Provide personalized style advice with chat history context"""
        wardrobe_context = await self.get_user_wardrobe_context(user, db)
        
        # User profile context
        user_context = f"""User Profile:
        • Gender: {user.gender.value if user.gender else 'not specified'}
        • Age: {user.age}
        • Height: {user.height}cm
        
        {wardrobe_context}
        
        Recent Conversation Context:
        {chat_history}
        """
        
        prompt = f"""{self.fashion_context}
        
        {user_context}
        
        User's current question: "{question}"
        
        Provide personalized fashion advice considering:
        1. Their existing wardrobe items (be specific about what they own)
        2. Their personal profile
        3. Recent conversation context (refer to previous questions/topics discussed)
        4. Current fashion trends
        5. Practical styling tips
        6. Color coordination
        7. Occasion-appropriate suggestions
        
        Keep your response concise, practical, and easy to read. Use simple formatting without asterisks.
        When referencing their wardrobe or previous conversation, be specific.
        If this is a follow-up question, acknowledge the previous context.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error providing style advice with history: {e}")
            return "I'm having trouble accessing my styling knowledge right now. Please try asking again!"
    
    def get_chat_history(self, user: User, db: Session, limit: int = 20) -> List[Dict]:
        """Get user's chat history"""
        messages = db.query(ChatMessage).filter(
            ChatMessage.user_id == user.id
        ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
        
        history = []
        for msg in reversed(messages):
            if msg.is_from_user:
                history.append({
                    "type": "user",
                    "message": msg.message,
                    "timestamp": msg.created_at.isoformat()
                })
            else:
                history.append({
                    "type": "ai",
                    "message": msg.response,
                    "timestamp": msg.created_at.isoformat(),
                    "message_type": msg.message_type
                })
        
        return history


# Global chatbot instance
chatbot = WardrobeChatbot()