import pydeck as pdk
from data_generator import ZONES

def build_map(urgency_df, map_view_mode="Urgency Plot", highlighted_zone=None):
    map_data = []
    for _, row in urgency_df.iterrows():
        zone = row['zone']
        meta = ZONES[zone]
        
        score = row['urgency_score']
        
        if map_view_mode == "AI Risk Clusters" and 'risk_profile' in row:
            profile = row['risk_profile']
            if profile == "Severe Outbreak":
                color = [147, 51, 234, 255]  # Purple
            elif profile == "Emerging Hotspot":
                color = [249, 115, 22, 255]  # Orange
            else:
                color = [56, 189, 248, 255]  # Light Blue
            tooltip_urgency = f"Profile: {profile}\nUrgency Score: {score:.3f}"
        else:
            if score > 0.5:
                color = [248, 113, 113, 255]  # Red
            elif score > 0.3:
                color = [251, 191, 36, 255]   # Yellow
            else:
                color = [74, 222, 128, 255]   # Green
            tooltip_urgency = f"{score:.3f}"
            
        radius = max(1500, score * 6000)
        
        # ── Handle Dynamic Chatbot Highlighting ──
        if highlighted_zone:
            if zone == highlighted_zone:
                # Highlight strongly: scale up radius and force high opacity
                radius = radius * 1.5
                color[3] = 255
            else:
                # Dim unhighlighted ones heavily
                color[3] = 40
        else:
            # Default normal opacity
            color[3] = 200

        map_data.append({
            "zone": zone,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "urgency": tooltip_urgency,
            "radius": radius, 
            "color": color
        })
        
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="color",
        pickable=True,
    )
    
    view_state = pdk.ViewState(
        latitude=13.0600,
        longitude=80.2450,
        zoom=10.5,
        pitch=45,
    )
    
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{zone}\nUrgency/Profile: {urgency}"},
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK
    )
