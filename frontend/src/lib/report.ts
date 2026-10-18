import type { ChatMessage } from './types';

export function downloadGeoJsonReport(title: string, messages: ChatMessage[]) {
  // Collect all features from all messages in the session
  const allFeatures: any[] = [];
  
  messages.forEach(msg => {
    if (msg.evidence?.geoJson) {
      const geo = msg.evidence.geoJson;
      // Add conversation context to the properties of each feature
      const metadata = {
        source_message_id: msg.id,
        timestamp: msg.timestamp,
        role: msg.role
      };

      if (geo.type === 'FeatureCollection' && Array.isArray(geo.features)) {
        const enrichedFeatures = geo.features.map((f: any) => ({
          ...f,
          properties: { ...f.properties, ...metadata }
        }));
        allFeatures.push(...enrichedFeatures);
      } else if (geo.type === 'Feature') {
        allFeatures.push({
          ...geo,
          properties: { ...geo.properties, ...metadata }
        });
      }
    }
  });

  if (allFeatures.length === 0) {
    alert('No geographic data (GeoJSON) found in this conversation to export.');
    return;
  }

  const featureCollection = {
    type: 'FeatureCollection',
    name: title || 'Diachron Export',
    metadata: {
      generatedAt: new Date().toISOString(),
      agent: 'Diachron AI'
    },
    features: allFeatures
  };

  // Trigger file download
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(featureCollection, null, 2));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  
  const safeTitle = title.replace(/[^a-z0-9]/gi, '_').toLowerCase() || 'diachron_export';
  downloadAnchorNode.setAttribute("download", `${safeTitle}.geojson`);
  
  document.body.appendChild(downloadAnchorNode); // required for firefox
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}
