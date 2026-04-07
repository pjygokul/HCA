import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Expanded to 25 major Chennai regions with real coordinates and realistic capacities
ZONES = {
    # Central Chennai
    "Adyar": {"lat": 13.0033, "lon": 80.2555, "pop": 250000, "icu": 50, "doc": 100, "o2": 200},
    "Anna Nagar": {"lat": 13.0836, "lon": 80.2110, "pop": 300000, "icu": 80, "doc": 150, "o2": 300},
    "Tondiarpet": {"lat": 13.1251, "lon": 80.2976, "pop": 200000, "icu": 30, "doc": 60, "o2": 150},
    "Mylapore": {"lat": 13.0336, "lon": 80.2743, "pop": 220000, "icu": 45, "doc": 90, "o2": 180},
    "Velachery": {"lat": 12.9750, "lon": 80.2212, "pop": 260000, "icu": 60, "doc": 110, "o2": 250},
    "T. Nagar": {"lat": 13.0405, "lon": 80.2337, "pop": 240000, "icu": 55, "doc": 105, "o2": 220},
    
    # North Chennai
    "Royapuram": {"lat": 13.1073, "lon": 80.2939, "pop": 180000, "icu": 35, "doc": 70, "o2": 140},
    "Perambur": {"lat": 13.1129, "lon": 80.2437, "pop": 190000, "icu": 40, "doc": 75, "o2": 160},
    "Ambattur": {"lat": 13.0982, "lon": 80.1597, "pop": 280000, "icu": 65, "doc": 120, "o2": 270},
    "Avadi": {"lat": 13.1147, "lon": 79.9910, "pop": 170000, "icu": 25, "doc": 50, "o2": 120},
    "Manali": {"lat": 13.1649, "lon": 80.2636, "pop": 160000, "icu": 20, "doc": 40, "o2": 100},
    "Thiruvottiyur": {"lat": 13.1581, "lon": 80.3010, "pop": 140000, "icu": 18, "doc": 35, "o2": 90},
    
    # South Chennai
    "Tambaram": {"lat": 12.9249, "lon": 80.1000, "pop": 320000, "icu": 70, "doc": 140, "o2": 290},
    "Chrompet": {"lat": 12.9516, "lon": 80.1462, "pop": 210000, "icu": 42, "doc": 80, "o2": 170},
    "Pallavaram": {"lat": 12.9675, "lon": 80.1491, "pop": 195000, "icu": 38, "doc": 75, "o2": 155},
    "Sholinganallur": {"lat": 12.9009, "lon": 80.2279, "pop": 180000, "icu": 45, "doc": 85, "o2": 175},
    "Thoraipakkam": {"lat": 12.9407, "lon": 80.2340, "pop": 150000, "icu": 30, "doc": 55, "o2": 125},
    "Medavakkam": {"lat": 12.9200, "lon": 80.1925, "pop": 130000, "icu": 25, "doc": 45, "o2": 105},
    
    # West Chennai
    "Kodambakkam": {"lat": 13.0533, "lon": 80.2265, "pop": 200000, "icu": 50, "doc": 95, "o2": 190},
    "Porur": {"lat": 13.0358, "lon": 80.1597, "pop": 175000, "icu": 35, "doc": 65, "o2": 145},
    "Koyambedu": {"lat": 13.0732, "lon": 80.1946, "pop": 165000, "icu": 32, "doc": 60, "o2": 135},
    "Vadapalani": {"lat": 13.0504, "lon": 80.2121, "pop": 185000, "icu": 40, "doc": 80, "o2": 165},
    
    # East Chennai
    "Thiruvanmiyur": {"lat": 12.9830, "lon": 80.2595, "pop": 170000, "icu": 35, "doc": 70, "o2": 150},
    "Besant Nagar": {"lat": 13.0067, "lon": 80.2669, "pop": 120000, "icu": 28, "doc": 55, "o2": 115},
    "Injambakkam": {"lat": 12.9167, "lon": 80.2500, "pop": 110000, "icu": 22, "doc": 42, "o2": 95},
}

RESOURCE_TOTALS = {
    "icu": sum(z["icu"] for z in ZONES.values()),
    "doc": sum(z["doc"] for z in ZONES.values()),
    "o2": sum(z["o2"] for z in ZONES.values())
}

def get_zone_metadata(zone):
    return ZONES.get(zone)

def generate_historical_data(days=180):
    np.random.seed(42)  # For reproducibility during demo
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date)
    
    records = []
    
    # Zone-specific base multipliers reflecting real Chennai demographics and risk factors
    zone_base_multiplier = {
        # Central Chennai - high density, mixed conditions
        "Adyar": 1.2, "Anna Nagar": 1.5, "Tondiarpet": 2.8, "Mylapore": 1.0, 
        "Velachery": 1.3, "T. Nagar": 1.4,
        
        # North Chennai - industrial, higher pollution, dense population
        "Royapuram": 2.2, "Perambur": 2.0, "Ambattur": 1.8, "Avadi": 1.6,
        "Manali": 2.5, "Thiruvottiyur": 2.3,
        
        # South Chennai - IT corridor, better infrastructure but growing density
        "Tambaram": 1.7, "Chrompet": 1.4, "Pallavaram": 1.5, "Sholinganallur": 1.3,
        "Thoraipakkam": 1.1, "Medavakkam": 1.2,
        
        # West Chennai - mixed residential and commercial
        "Kodambakkam": 1.6, "Porur": 1.3, "Koyambedu": 1.9, "Vadapalani": 1.4,
        
        # East Chennai - coastal, varied socioeconomic conditions
        "Thiruvanmiyur": 1.1, "Besant Nagar": 0.9, "Injambakkam": 1.0,
    }
    
    for zone, meta in ZONES.items():
        base_cases = np.random.randint(5, 12) * zone_base_multiplier[zone]
        
        # Enhanced seasonal trend simulation
        t = np.arange(days)
        
        # Northeast monsoon peak (Oct-Jan)
        monsoon_peak = 20 * np.sin(2 * np.pi * (t + 90) / 365) * np.exp(-(((t % 365) - 300)**2) / (2 * 50**2))
        
        # Summer heat stress peak (Apr-Jun)  
        summer_peak = 12 * np.sin(2 * np.pi * (t + 180) / 365) * np.exp(-(((t % 365) - 120)**2) / (2 * 40**2))
        
        # Weekly cycle (weekends lower)
        weekly_cycle = 3 * np.sin(2 * np.pi * t / 7)
        
        # Random noise and occasional spikes
        noise = np.random.normal(0, 3, days)
        spikes = np.random.poisson(0.02, days) * np.random.exponential(15, days)  # Occasional outbreaks
        
        # Combine all patterns
        seasonal_pattern = monsoon_peak + summer_peak + weekly_cycle
        cases = np.maximum(0, base_cases + seasonal_pattern + noise + spikes).astype(int)
        
        # Add some realistic variance based on zone characteristics
        if zone in ["Tondiarpet", "Manali", "Thiruvottiyur"]:  # Industrial areas
            cases = np.maximum(cases, cases * np.random.uniform(1.1, 1.3, days)).astype(int)
        elif zone in ["Besant Nagar", "Thoraipakkam"]:  # Affluent areas
            cases = (cases * np.random.uniform(0.7, 0.9, days)).astype(int)
        
        for i, dt in enumerate(dates):
            records.append({
                "date": dt,
                "zone": zone,
                "cases": cases[i],
                "pop": meta["pop"],
                "icu_capacity": meta["icu"],
                "doctors_capacity": meta["doc"],
                "oxygen_capacity": meta["o2"]
            })
    
    return pd.DataFrame(records)

# Summary statistics
def get_city_stats():
    total_pop = sum(z["pop"] for z in ZONES.values())
    total_icu = sum(z["icu"] for z in ZONES.values())
    total_docs = sum(z["doc"] for z in ZONES.values())
    total_o2 = sum(z["o2"] for z in ZONES.values())
    
    return {
        "total_zones": len(ZONES),
        "total_population": total_pop,
        "total_icu_beds": total_icu,
        "total_doctors": total_docs,
        "total_oxygen_units": total_o2,
        "avg_pop_per_zone": total_pop // len(ZONES),
        "icu_per_100k": round(total_icu / (total_pop / 100000), 1),
        "docs_per_100k": round(total_docs / (total_pop / 100000), 1)
    }

# Test the data generation
if __name__ == "__main__":
    print("Chennai Health Surveillance - Enhanced Zone Coverage")
    print("=" * 55)
    
    stats = get_city_stats()
    print(f"Total Zones: {stats['total_zones']}")
    print(f"Total Population: {stats['total_population']:,}")
    print(f"Total ICU Beds: {stats['total_icu_beds']}")
    print(f"Total Doctors: {stats['total_doctors']}")
    print(f"Total Oxygen Units: {stats['total_oxygen_units']}")
    print(f"ICU Beds per 100k: {stats['icu_per_100k']}")
    print(f"Doctors per 100k: {stats['docs_per_100k']}")
    
    print("\nGenerating sample data...")
    df = generate_historical_data(30)
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Case range: {df['cases'].min()} to {df['cases'].max()}")
    print("\nTop 5 zones by average cases:")
    avg_cases = df.groupby('zone')['cases'].mean().sort_values(ascending=False).head()
    for zone, cases in avg_cases.items():
        print(f"  {zone}: {cases:.1f} cases/day")
