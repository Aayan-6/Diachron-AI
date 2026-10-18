/**
 * Diachron AI Geocoding & Geographic Search Service
 * Supports worldwide cities, states, countries, landmarks, and raw coordinates.
 * Dynamically scales zoom level based on place hierarchy:
 *   - City: Zoom 12.0 - 13.5 (high-detail urban street grid)
 *   - State / Province: Zoom 7.0 - 8.5 (regional boundary overview)
 *   - Country: Zoom 4.0 - 5.5 (continental / national overview)
 *   - Landmark / POI: Zoom 14.5 - 16.5 (close-up feature analysis)
 */

export type LocationCategory = 'country' | 'state' | 'city' | 'poi';

export interface GeocodingResult {
  id: string;
  name: string;
  displayName: string;
  category: LocationCategory;
  center: [number, number]; // [lng, lat]
  bounds?: [[number, number], [number, number]]; // [[minLng, minLat], [maxLng, maxLat]]
  zoom: number;
  badgeLabel: string;
}

// Built-in Offline Fallback Database of Major World Countries, States, and Global Cities
const FALLBACK_LOCATIONS: GeocodingResult[] = [
  // --- COUNTRIES (Zoom 4.0 - 5.5) ---
  {
    id: 'country-in',
    name: 'India',
    displayName: 'India (Republic of India)',
    category: 'country',
    center: [78.9629, 20.5937],
    bounds: [[68.16, 6.75], [97.40, 35.50]],
    zoom: 4.8,
    badgeLabel: 'Country'
  },
  {
    id: 'country-us',
    name: 'United States',
    displayName: 'United States of America',
    category: 'country',
    center: [-98.5795, 39.8283],
    bounds: [[-125.0, 24.5], [-66.9, 49.4]],
    zoom: 4.2,
    badgeLabel: 'Country'
  },
  {
    id: 'country-gb',
    name: 'United Kingdom',
    displayName: 'United Kingdom',
    category: 'country',
    center: [-3.4360, 55.3781],
    bounds: [[-8.6, 49.9], [1.8, 60.9]],
    zoom: 5.5,
    badgeLabel: 'Country'
  },
  {
    id: 'country-jp',
    name: 'Japan',
    displayName: 'Japan',
    category: 'country',
    center: [138.2529, 36.2048],
    bounds: [[129.5, 31.0], [145.8, 45.5]],
    zoom: 5.2,
    badgeLabel: 'Country'
  },
  {
    id: 'country-de',
    name: 'Germany',
    displayName: 'Germany',
    category: 'country',
    center: [10.4515, 51.1657],
    bounds: [[5.9, 47.3], [15.0, 55.1]],
    zoom: 5.8,
    badgeLabel: 'Country'
  },
  {
    id: 'country-fr',
    name: 'France',
    displayName: 'France',
    category: 'country',
    center: [2.2137, 46.2276],
    bounds: [[-4.8, 41.3], [9.6, 51.1]],
    zoom: 5.5,
    badgeLabel: 'Country'
  },
  {
    id: 'country-au',
    name: 'Australia',
    displayName: 'Australia',
    category: 'country',
    center: [133.7751, -25.2744],
    bounds: [[113.3, -43.6], [153.6, -10.7]],
    zoom: 4.0,
    badgeLabel: 'Country'
  },
  {
    id: 'country-ca',
    name: 'Canada',
    displayName: 'Canada',
    category: 'country',
    center: [-106.3468, 56.1304],
    bounds: [[-141.0, 41.7], [-52.6, 83.1]],
    zoom: 3.8,
    badgeLabel: 'Country'
  },
  {
    id: 'country-br',
    name: 'Brazil',
    displayName: 'Brazil',
    category: 'country',
    center: [-51.9253, -14.2350],
    bounds: [[-73.9, -33.7], [-34.8, 5.3]],
    zoom: 4.2,
    badgeLabel: 'Country'
  },
  {
    id: 'country-ae',
    name: 'United Arab Emirates',
    displayName: 'United Arab Emirates',
    category: 'country',
    center: [54.3773, 24.4539],
    bounds: [[51.6, 22.6], [56.4, 26.1]],
    zoom: 7.2,
    badgeLabel: 'Country'
  },
  {
    id: 'country-sg',
    name: 'Singapore',
    displayName: 'Republic of Singapore',
    category: 'country',
    center: [103.8198, 1.3521],
    bounds: [[103.6, 1.15], [104.1, 1.48]],
    zoom: 11.5,
    badgeLabel: 'Country'
  },

  // --- STATES & PROVINCES (Zoom 6.8 - 8.5) ---
  {
    id: 'state-karnataka',
    name: 'Karnataka',
    displayName: 'Karnataka, India',
    category: 'state',
    center: [75.7139, 15.3173],
    bounds: [[74.05, 11.59], [78.58, 18.47]],
    zoom: 7.5,
    badgeLabel: 'State'
  },
  {
    id: 'state-maharashtra',
    name: 'Maharashtra',
    displayName: 'Maharashtra, India',
    category: 'state',
    center: [75.7139, 19.7515],
    bounds: [[72.6, 15.6], [80.9, 22.0]],
    zoom: 7.2,
    badgeLabel: 'State'
  },
  {
    id: 'state-delhi-ncr',
    name: 'Delhi NCR',
    displayName: 'National Capital Region (Delhi), India',
    category: 'state',
    center: [77.1025, 28.7041],
    bounds: [[76.8, 28.4], [77.4, 28.9]],
    zoom: 10.5,
    badgeLabel: 'State / NCR'
  },
  {
    id: 'state-tamil-nadu',
    name: 'Tamil Nadu',
    displayName: 'Tamil Nadu, India',
    category: 'state',
    center: [78.6569, 11.1271],
    bounds: [[76.2, 8.1], [80.3, 13.6]],
    zoom: 7.5,
    badgeLabel: 'State'
  },
  {
    id: 'state-uttar-pradesh',
    name: 'Uttar Pradesh',
    displayName: 'Uttar Pradesh, India',
    category: 'state',
    center: [80.9462, 26.8467],
    bounds: [[77.1, 23.9], [84.6, 30.4]],
    zoom: 7.0,
    badgeLabel: 'State'
  },
  {
    id: 'state-rajasthan',
    name: 'Rajasthan',
    displayName: 'Rajasthan, India',
    category: 'state',
    center: [74.2179, 27.0238],
    bounds: [[69.5, 23.1], [78.3, 30.2]],
    zoom: 6.8,
    badgeLabel: 'State'
  },
  {
    id: 'state-kerala',
    name: 'Kerala',
    displayName: 'Kerala, India',
    category: 'state',
    center: [76.2711, 10.8505],
    bounds: [[74.8, 8.3], [77.4, 12.8]],
    zoom: 7.8,
    badgeLabel: 'State'
  },
  {
    id: 'state-gujarat',
    name: 'Gujarat',
    displayName: 'Gujarat, India',
    category: 'state',
    center: [71.1924, 22.2587],
    bounds: [[68.1, 20.1], [74.5, 24.7]],
    zoom: 7.2,
    badgeLabel: 'State'
  },
  {
    id: 'state-california',
    name: 'California',
    displayName: 'California, United States',
    category: 'state',
    center: [-119.4179, 36.7783],
    bounds: [[-124.4, 32.5], [-114.1, 42.0]],
    zoom: 6.8,
    badgeLabel: 'State'
  },
  {
    id: 'state-texas',
    name: 'Texas',
    displayName: 'Texas, United States',
    category: 'state',
    center: [-99.9018, 31.9686],
    bounds: [[-106.6, 25.8], [-93.5, 36.5]],
    zoom: 6.2,
    badgeLabel: 'State'
  },
  {
    id: 'state-new-york',
    name: 'New York (State)',
    displayName: 'New York State, United States',
    category: 'state',
    center: [-75.5268, 43.2994],
    bounds: [[-79.8, 40.5], [-71.9, 45.0]],
    zoom: 6.8,
    badgeLabel: 'State'
  },
  {
    id: 'state-bavaria',
    name: 'Bavaria',
    displayName: 'Bavaria, Germany',
    category: 'state',
    center: [11.4979, 48.7904],
    bounds: [[8.9, 47.2], [13.9, 50.6]],
    zoom: 7.5,
    badgeLabel: 'State'
  },

  // --- CITIES (Zoom 12.0 - 13.5) ---
  {
    id: 'city-bengaluru',
    name: 'Bengaluru',
    displayName: 'Bengaluru, Karnataka, India',
    category: 'city',
    center: [77.5946, 12.9716],
    bounds: [[77.46, 12.83], [77.78, 13.14]],
    zoom: 12.8,
    badgeLabel: 'City'
  },
  {
    id: 'city-delhi',
    name: 'Delhi',
    displayName: 'New Delhi, Delhi, India',
    category: 'city',
    center: [77.2090, 28.6139],
    bounds: [[77.05, 28.50], [77.35, 28.75]],
    zoom: 12.5,
    badgeLabel: 'Capital City'
  },
  {
    id: 'city-mumbai',
    name: 'Mumbai',
    displayName: 'Mumbai, Maharashtra, India',
    category: 'city',
    center: [72.8777, 19.0760],
    bounds: [[72.75, 18.89], [72.98, 19.27]],
    zoom: 12.2,
    badgeLabel: 'City'
  },
  {
    id: 'city-chennai',
    name: 'Chennai',
    displayName: 'Chennai, Tamil Nadu, India',
    category: 'city',
    center: [80.2707, 13.0827],
    bounds: [[80.15, 12.95], [80.32, 13.20]],
    zoom: 12.5,
    badgeLabel: 'City'
  },
  {
    id: 'city-kolkata',
    name: 'Kolkata',
    displayName: 'Kolkata, West Bengal, India',
    category: 'city',
    center: [88.3639, 22.5726],
    bounds: [[88.25, 22.45], [88.45, 22.65]],
    zoom: 12.5,
    badgeLabel: 'City'
  },
  {
    id: 'city-hyderabad',
    name: 'Hyderabad',
    displayName: 'Hyderabad, Telangana, India',
    category: 'city',
    center: [78.4867, 17.3850],
    bounds: [[78.35, 17.25], [78.60, 17.50]],
    zoom: 12.5,
    badgeLabel: 'City'
  },
  {
    id: 'city-pune',
    name: 'Pune',
    displayName: 'Pune, Maharashtra, India',
    category: 'city',
    center: [73.8567, 18.5204],
    bounds: [[73.75, 18.42], [73.98, 18.62]],
    zoom: 12.5,
    badgeLabel: 'City'
  },
  {
    id: 'city-mangalore',
    name: 'Mangalore',
    displayName: 'Mangaluru (SAR Port), Karnataka, India',
    category: 'city',
    center: [74.8560, 12.9141],
    bounds: [[74.78, 12.85], [74.92, 12.98]],
    zoom: 13.0,
    badgeLabel: 'Coastal City'
  },
  {
    id: 'city-tokyo',
    name: 'Tokyo',
    displayName: 'Tokyo, Japan',
    category: 'city',
    center: [139.6917, 35.6895],
    bounds: [[139.55, 35.55], [139.85, 35.80]],
    zoom: 12.2,
    badgeLabel: 'Metropolis'
  },
  {
    id: 'city-paris',
    name: 'Paris',
    displayName: 'Paris, Île-de-France, France',
    category: 'city',
    center: [2.3522, 48.8566],
    bounds: [[2.22, 48.81], [2.47, 48.90]],
    zoom: 12.8,
    badgeLabel: 'City'
  },
  {
    id: 'city-london',
    name: 'London',
    displayName: 'London, Greater London, United Kingdom',
    category: 'city',
    center: [-0.1278, 51.5074],
    bounds: [[-0.35, 51.38], [0.15, 51.67]],
    zoom: 12.2,
    badgeLabel: 'City'
  },
  {
    id: 'city-new-york-city',
    name: 'New York City',
    displayName: 'New York City, New York, United States',
    category: 'city',
    center: [-74.0060, 40.7128],
    bounds: [[-74.26, 40.49], [-73.70, 40.92]],
    zoom: 12.2,
    badgeLabel: 'City'
  },
  {
    id: 'city-san-francisco',
    name: 'San Francisco',
    displayName: 'San Francisco, California, United States',
    category: 'city',
    center: [-122.4194, 37.7749],
    bounds: [[-122.52, 37.70], [-122.35, 37.83]],
    zoom: 13.0,
    badgeLabel: 'City'
  },
  {
    id: 'city-dubai',
    name: 'Dubai',
    displayName: 'Dubai, United Arab Emirates',
    category: 'city',
    center: [55.2708, 25.2048],
    bounds: [[55.10, 25.00], [55.45, 25.35]],
    zoom: 12.2,
    badgeLabel: 'City'
  },
  {
    id: 'city-sydney',
    name: 'Sydney',
    displayName: 'Sydney, New South Wales, Australia',
    category: 'city',
    center: [151.2093, -33.8688],
    bounds: [[151.05, -33.95], [151.30, -33.75]],
    zoom: 12.2,
    badgeLabel: 'City'
  }
];

/**
 * Checks if the query represents raw coordinates e.g. "28.6139, 77.2090"
 */
export function parseCoordinates(query: string): GeocodingResult | null {
  const q = query.trim();
  const match = q.match(/^([-+]?\d+(?:\.\d+)?)[,\s]+([-+]?\d+(?:\.\d+)?)$/);
  if (!match) return null;

  const num1 = parseFloat(match[1]);
  const num2 = parseFloat(match[2]);

  // Determine which is lat (-90 to 90) and which is lng (-180 to 180)
  let lat: number, lng: number;
  if (Math.abs(num1) <= 90 && Math.abs(num2) <= 180) {
    lat = num1;
    lng = num2;
  } else if (Math.abs(num2) <= 90 && Math.abs(num1) <= 180) {
    lat = num2;
    lng = num1;
  } else {
    return null;
  }

  return {
    id: `coords-${lat}-${lng}`,
    name: `${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E`,
    displayName: `Geographic Point Coordinates: [${lat.toFixed(5)}, ${lng.toFixed(5)}]`,
    category: 'poi',
    center: [lng, lat],
    zoom: 13.5,
    badgeLabel: 'Coordinates'
  };
}

/**
 * Classifies Nominatim result into country, state, city, or poi.
 */
function classifyNominatimType(item: any): { category: LocationCategory; badgeLabel: string; zoom: number } {
  const rawType = (item.addresstype || item.type || '').toLowerCase();
  const address = item.address || {};

  if (rawType === 'country' || item.class === 'boundary' && item.type === 'administrative' && address.country && !address.state && !address.city) {
    return { category: 'country', badgeLabel: 'Country', zoom: 4.8 };
  }

  if (['state', 'province', 'region', 'state_district', 'territory', 'prefecture'].includes(rawType)) {
    return { category: 'state', badgeLabel: 'State / Province', zoom: 7.5 };
  }

  if (['city', 'town', 'municipality', 'county', 'district', 'suburb', 'borough'].includes(rawType) || address.city || address.town) {
    return { category: 'city', badgeLabel: 'City / Urban', zoom: 12.5 };
  }

  return { category: 'poi', badgeLabel: 'Location / Landmark', zoom: 14.5 };
}

/**
 * Executes a geographic query against Nominatim with automatic fallback.
 * Guarantees that:
 *   - City search -> Zoomed in more (~12-13.5)
 *   - State search -> Medium zoom (~7-8.5)
 *   - Country search -> Broad zoom (~4-5.5)
 */
export async function searchLocations(query: string, signal?: AbortSignal): Promise<GeocodingResult[]> {
  const trimmed = query.trim();
  if (!trimmed || trimmed.length < 2) return [];

  // 1. Check coordinates first
  const coordResult = parseCoordinates(trimmed);
  if (coordResult) return [coordResult];

  // 2. Query OpenStreetMap Nominatim with a fast timeout
  try {
    const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
      trimmed
    )}&limit=5&addressdetails=1`;

    const res = await fetch(url, {
      signal,
      headers: {
        'Accept': 'application/json',
      }
    });

    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) {
        const results: GeocodingResult[] = data.map((item: any) => {
          const { category, badgeLabel, zoom } = classifyNominatimType(item);

          let bounds: [[number, number], [number, number]] | undefined = undefined;
          if (Array.isArray(item.boundingbox) && item.boundingbox.length === 4) {
            const minLat = parseFloat(item.boundingbox[0]);
            const maxLat = parseFloat(item.boundingbox[1]);
            const minLon = parseFloat(item.boundingbox[2]);
            const maxLon = parseFloat(item.boundingbox[3]);

            if (!isNaN(minLat) && !isNaN(maxLat) && !isNaN(minLon) && !isNaN(maxLon)) {
              bounds = [[minLon, minLat], [maxLon, maxLat]];
            }
          }

          const lat = parseFloat(item.lat);
          const lon = parseFloat(item.lon);

          return {
            id: `nom-${item.place_id || Math.random()}`,
            name: item.name || item.display_name?.split(',')[0] || trimmed,
            displayName: item.display_name || trimmed,
            category,
            center: [lon, lat],
            bounds,
            zoom,
            badgeLabel
          };
        });

        return results;
      }
    }
  } catch {
    // Network failure or abort -> fall through to local database
  }

  // 3. Fallback: Search built-in offline database
  const lower = trimmed.toLowerCase();
  const matched = FALLBACK_LOCATIONS.filter(
    (loc) =>
      loc.name.toLowerCase().includes(lower) ||
      loc.displayName.toLowerCase().includes(lower)
  );

  return matched.slice(0, 5);
}
