"""
Interactive GIS Mapping & Analytical Plotting Module
Implements:
- Tab 1: Data Ingestion Map (Tower Markers + Drive Test Measurements Route)
- Tab 3: 100m x 100m Rectangle Polygon Coverage Area Map with 4 DVB-T2 Tiers (Blank Spot Transparent)
- Multi-provider Tile Layers (CartoDB Positron, OpenStreetMap, Esri Satellite)
- Plotly Parity Plot and Feature Importance Visualization
"""

import folium
from folium.plugins import FastMarkerCluster
from branca.element import MacroElement, Template
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class RasterCoverageInspector(MacroElement):
    """
    Client-Side Interactive Hover Inspector for Continuous Telecom Raster Heatmaps:
    - Executed natively by streamlit_folium inside React/Leaflet script context.
    - Directly binds to the Leaflet map instance (map_div).
    - Features real-time hover tooltip matching the previous vector grid box model:
      '● Sangat Kuat: 68.4 dBµV/m | d=3.42 km | Clr=+18.5m'
    - Highlights the exact grid cell under the cursor.
    - Embeds a native Leaflet HUD bottom-left telemetry control.
    - Supports click popup with full station telemetry.
    """
    def __init__(self, raster_data: Dict[str, Any]):
        super().__init__()
        self._name = "RasterCoverageInspector"
        self.raster_data = raster_data

        template_str = """
            {% macro script(this, kwargs) %}
                var map = {{ this._parent.get_name() }};

                // 1. Inject custom styling into document
                var customStyle = document.createElement('style');
                customStyle.innerHTML = `
                    .cloudrf-grid-tooltip {
                        background: rgba(15, 23, 42, 0.95) !important;
                        color: #f8fafc !important;
                        border: 1px solid rgba(255, 255, 255, 0.25) !important;
                        border-radius: 6px !important;
                        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6) !important;
                        padding: 5px 9px !important;
                        pointer-events: none !important;
                    }
                    .cloudrf-grid-tooltip:before {
                        border-top-color: rgba(15, 23, 42, 0.95) !important;
                    }
                `;
                document.head.appendChild(customStyle);

                // 2. Grid metadata & byte array decoding
                var H = __H__;
                var W = __W__;
                var minLat = __MIN_LAT__;
                var maxLat = __MAX_LAT__;
                var minLon = __MIN_LON__;
                var maxLon = __MAX_LON__;
                var txLat = __TX_LAT__;
                var txLon = __TX_LON__;
                var txName = "__TX_NAME__";
                var freqMhz = __FREQ_MHZ__;
                var resM = __RES_M__;

                var stepLat = (maxLat - minLat) / H;
                var stepLon = (maxLon - minLon) / W;

                var fsRaw = atob("__FS_B64__");
                var fsBytes = new Uint8Array(fsRaw.length);
                for (var i = 0; i < fsRaw.length; i++) fsBytes[i] = fsRaw.charCodeAt(i);

                var clrRaw = atob("__CLR_B64__");
                var clrBytes = new Uint8Array(clrRaw.length);
                for (var j = 0; j < clrRaw.length; j++) clrBytes[j] = clrRaw.charCodeAt(j);

                var elevRaw = atob("__ELEV_B64__");
                var elevBytes = new Uint8Array(elevRaw.length);
                for (var k = 0; k < elevRaw.length; k++) elevBytes[k] = elevRaw.charCodeAt(k);
                var elevGrid = new Int16Array(elevBytes.buffer);

                var probeTooltip = L.tooltip({
                    sticky: false,
                    direction: 'top',
                    offset: [0, -10],
                    opacity: 0.98,
                    className: 'cloudrf-grid-tooltip'
                });

                var cursorBox = null;

                function haversineDist(lat1, lon1, lat2, lon2) {
                    var R = 6371.0;
                    var dLat = (lat2 - lat1) * Math.PI / 180.0;
                    var dLon = (lon2 - lon1) * Math.PI / 180.0;
                    var a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                            Math.cos(lat1 * Math.PI / 180.0) * Math.cos(lat2 * Math.PI / 180.0) *
                            Math.sin(dLon/2) * Math.sin(dLon/2);
                    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
                }

                function clearInspector() {
                    if (map.hasLayer(probeTooltip)) map.removeLayer(probeTooltip);
                    if (cursorBox && map.hasLayer(cursorBox)) map.removeLayer(cursorBox);
                }

                var mapContainer = map.getContainer();
                if (mapContainer) {
                    mapContainer.addEventListener('mouseover', function(ev) {
                        if (ev.target && ev.target.closest && ev.target.closest('.leaflet-marker-icon, .leaflet-popup, .leaflet-tooltip')) {
                            clearInspector();
                        }
                    }, true);
                }

                map.on('mousemove', function(e) {
                    if (e.originalEvent && e.originalEvent.target) {
                        var tgt = e.originalEvent.target;
                        if (tgt.closest && tgt.closest('.leaflet-marker-icon, .leaflet-popup, .leaflet-control, .leaflet-tooltip')) {
                            clearInspector();
                            return;
                        }
                    }

                    var lat = e.latlng.lat;
                    var lon = e.latlng.lng;

                    if (lat < minLat || lat > maxLat || lon < minLon || lon > maxLon) {
                        clearInspector();
                        return;
                    }

                    var r = Math.floor((maxLat - lat) / stepLat);
                    var c = Math.floor((lon - minLon) / stepLon);

                    if (r < 0 || r >= H || c < 0 || c >= W) {
                        clearInspector();
                        return;
                    }

                    var idx = r * W + c;
                    var rawFs = fsBytes[idx];

                    if (rawFs === 0) {
                        clearInspector();
                        return;
                    }

                    var fs = rawFs / 2.0;
                    var rawClr = clrBytes[idx];
                    var clr = rawClr > 0 ? (rawClr - 100) : null;
                    var elevM = (elevGrid && elevGrid.length > idx) ? elevGrid[idx] : null;
                    var d_km = haversineDist(lat, lon, txLat, txLon);

                    var cat = 'Blank Spot';
                    var badgeColor = '#94a3b8';
                    var isEst = (fs < 45.2 || (clr !== null && clr < 0));
                    var fsSuffix = isEst ? ' (Est.)' : '';

                    if (fs >= 65.0) {
                        cat = 'Sangat Kuat';
                        badgeColor = '#ef4444';
                    } else if (fs >= 54.0) {
                        cat = 'Kuat';
                        badgeColor = '#f97316';
                    } else if (fs >= 45.2) {
                        cat = 'Sedang';
                        badgeColor = '#3b82f6';
                    }

                    var clrStr = (clr !== null) ? ' | Clr=' + (clr >= 0 ? '+' : '') + clr.toFixed(1) + 'm' : '';
                    var elevStr = (elevM !== null) ? ' | Elev=' + elevM + 'm' : '';

                    var tooltipHtml = '<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">' +
                        '<span style="color:' + badgeColor + '; font-weight: 800;">● <b>' + cat + '</b></span>: ' +
                        '<span style="font-weight: 700; color: #ffffff;">' + fs.toFixed(1) + ' dBµV/m' + fsSuffix + '</span> | ' +
                        '<span style="color:#cbd5e1;">d=' + d_km.toFixed(2) + ' km</span>' +
                        '<span style="color:#34d399;">' + elevStr + '</span>' +
                        '<span style="color:#38bdf8;">' + clrStr + '</span>' +
                        '</div>';

                    probeTooltip.setLatLng(e.latlng);
                    probeTooltip.setContent(tooltipHtml);
                    if (!map.hasLayer(probeTooltip)) {
                        probeTooltip.addTo(map);
                    }

                    var cLatMax = maxLat - r * stepLat;
                    var cLatMin = maxLat - (r + 1) * stepLat;
                    var cLonMin = minLon + c * stepLon;
                    var cLonMax = minLon + (c + 1) * stepLon;

                    if (!cursorBox) {
                        cursorBox = L.rectangle(
                            [[cLatMin, cLonMin], [cLatMax, cLonMax]],
                            { color: '#38bdf8', weight: 1.5, fill: true, fillOpacity: 0.18, interactive: false }
                        ).addTo(map);
                    } else {
                        cursorBox.setBounds([[cLatMin, cLonMin], [cLatMax, cLonMax]]);
                        if (!map.hasLayer(cursorBox)) {
                            cursorBox.addTo(map);
                        }
                    }
                });

                map.on('mouseout', clearInspector);

                map.on('click', function(e) {
                    var lat = e.latlng.lat;
                    var lon = e.latlng.lng;
                    if (lat < minLat || lat > maxLat || lon < minLon || lon > maxLon) return;
                    var r = Math.floor((maxLat - lat) / stepLat);
                    var c = Math.floor((lon - minLon) / stepLon);
                    if (r < 0 || r >= H || c < 0 || c >= W) return;
                    var idx = r * W + c;
                    var rawFs = fsBytes[idx];
                    if (rawFs === 0) return;
                    var fs = rawFs / 2.0;
                    var rawClr = clrBytes[idx];
                    var clr = rawClr > 0 ? (rawClr - 100) : null;
                    var elevM = (elevGrid && elevGrid.length > idx) ? elevGrid[idx] : null;
                    var d_km = haversineDist(lat, lon, txLat, txLon);
                    var cat = fs >= 65.0 ? 'Sangat Kuat' : (fs >= 54.0 ? 'Kuat' : (fs >= 45.2 ? 'Sedang' : 'Blank Spot'));
                    var color = fs >= 65.0 ? '#ef4444' : (fs >= 54.0 ? '#f97316' : (fs >= 45.2 ? '#3b82f6' : '#94a3b8'));

                    var popupContent = '<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px; line-height: 1.5; min-width: 230px;">' +
                        '<h4 style="margin: 0 0 6px 0; color: #0284c7; font-size: 14px; font-weight: bold;">📊 Detail Sinyal Titik Grid</h4>' +
                        '<div style="border-bottom: 1px solid #cbd5e1; margin-bottom: 6px;"></div>' +
                        '<b>Pemancar:</b> ' + txName + ' (' + freqMhz + ' MHz)<br/>' +
                        '<b>Kuat Medan AI:</b> <span style="font-size: 15px; font-weight: bold; color: ' + color + ';">' + fs.toFixed(2) + ' dBµV/m</span><br/>' +
                        '<b>Kategori Layanan:</b> <b>' + cat + '</b><br/>' +
                        '<b>Jarak ke TX:</b> ' + d_km.toFixed(3) + ' km<br/>' +
                        '<b>Ketinggian Lokasi:</b> ' + (elevM !== null ? (elevM + ' mdpl') : 'N/A') + '<br/>' +
                        '<b>Clearance Topografi:</b> ' + (clr !== null ? ((clr >= 0 ? '+' : '') + clr.toFixed(2) + ' m') : 'N/A') + '<br/>' +
                        '<b>Koordinat Titik:</b> ' + lat.toFixed(6) + ', ' + lon.toFixed(6) + '<br/>' +
                        '<b>Resolusi Grid:</b> ' + resM + 'm × ' + resM + '<br/>' +
                        '</div>';

                    L.popup().setLatLng(e.latlng).setContent(popupContent).openOn(map);
                });
            {% endmacro %}
        """

        replacements = {
            "__H__": str(raster_data.get('H', 200)),
            "__W__": str(raster_data.get('W', 200)),
            "__MIN_LAT__": str(raster_data.get('min_lat', 0.0)),
            "__MAX_LAT__": str(raster_data.get('max_lat', 0.0)),
            "__MIN_LON__": str(raster_data.get('min_lon', 0.0)),
            "__MAX_LON__": str(raster_data.get('max_lon', 0.0)),
            "__TX_LAT__": str(raster_data.get('tx_lat', 0.0)),
            "__TX_LON__": str(raster_data.get('tx_lon', 0.0)),
            "__TX_NAME__": str(raster_data.get('tx_name', 'TX')).replace('"', '\\"'),
            "__FREQ_MHZ__": str(raster_data.get('freq_mhz', 0.0)),
            "__RES_M__": str(int(raster_data.get('resolution_m', 50))),
            "__FS_B64__": str(raster_data.get('fs_grid_b64', '')),
            "__CLR_B64__": str(raster_data.get('clr_grid_b64', '')),
            "__ELEV_B64__": str(raster_data.get('elev_grid_b64', ''))
        }
        for k, v in replacements.items():
            template_str = template_str.replace(k, v)
        self._template = Template(template_str)


class LeafletInteractiveZoomHelper(MacroElement):
    """
    Enables and guarantees high responsiveness of mouse wheel scroll zooming,
    smooth zoom gestures, and double-click zoom across desktop browsers and Streamlit iframes.
    """
    def __init__(self):
        super().__init__()
        self._name = "LeafletInteractiveZoomHelper"
        self._template = Template("""
            {% macro script(this, kwargs) %}
                var map = {{ this._parent.get_name() }};
                if (map) {
                    if (map.scrollWheelZoom) {
                        map.scrollWheelZoom.enable();
                    }
                    if (map.doubleClickZoom) {
                        map.doubleClickZoom.enable();
                    }
                    if (map.touchZoom) {
                        map.touchZoom.enable();
                    }
                    if (map.boxZoom) {
                        map.boxZoom.enable();
                    }
                    map.options.scrollWheelZoom = true;
                    map.options.wheelDebounceTime = 40;
                    map.options.wheelPxPerZoomLevel = 60;

                    var container = map.getContainer();
                    if (container) {
                        if (!document.getElementById('kml-poly-style')) {
                            var style = document.createElement('style');
                            style.id = 'kml-poly-style';
                            style.innerHTML = `
                                .kml-service-area-poly { pointer-events: none !important; }
                                .cloudrf-grid-tooltip {
                                    background: rgba(15, 23, 42, 0.95) !important;
                                    color: #f8fafc !important;
                                    border: 1px solid rgba(255, 255, 255, 0.25) !important;
                                    border-radius: 6px !important;
                                    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6) !important;
                                    padding: 5px 9px !important;
                                    pointer-events: none !important;
                                    white-space: nowrap !important;
                                }
                                .cloudrf-grid-tooltip:before {
                                    border-top-color: rgba(15, 23, 42, 0.95) !important;
                                }
                            `;
                            document.head.appendChild(style);
                        }
                        container.addEventListener('mouseenter', function() {
                            if (map.scrollWheelZoom) {
                                map.scrollWheelZoom.enable();
                            }
                        });
                        container.addEventListener('wheel', function(e) {
                            if (map.scrollWheelZoom) {
                                map.scrollWheelZoom.enable();
                            }
                            e.stopPropagation();
                        }, { passive: false });
                    }
                }
            {% endmacro %}
        """)


def create_cloudrf_gis_map(
    df_tx_summary: Optional[pd.DataFrame] = None,
    df_matched_dt: Optional[pd.DataFrame] = None,
    df_grid_result: Optional[pd.DataFrame] = None,
    raster_overlay_data: Optional[Dict[str, Any]] = None,
    selected_tx_name: Optional[str] = None,
    show_transmitters: bool = True,
    show_drive_test: bool = False,
    show_coverage_grid: bool = False,
    single_point_ctx: Optional[Dict[str, Any]] = None,
    single_point_res: Optional[Dict[str, Any]] = None,
    service_area_polygon: Optional[Dict[str, Any]] = None,
    service_area_testpoints: Optional[List[Dict[str, Any]]] = None,
    test_point_evaluations: Optional[pd.DataFrame] = None,
    basemap_choice: str = "satellite"
) -> folium.Map:
    """
    Master CloudRF/SOOTHSAYER GIS Map Engine:
    - Persistent top-level map canvas with professional RF telemetry look
    - Basemap options: Esri World Imagery (Satellite), CartoDB Dark Matter, OSM, Positron
    - Multi-layer capabilities: Transmitter towers, drive test routes, 100m polygon grid, and point rays
    - Official DVB-T2 Service Area Boundary (Permenkominfo KML Polygon) & Test Points
    - Vertical CloudRF gradient colorbar legend pinned on the top right
    """
    # 1. Determine map center & default zoom
    center_lat, center_lon = -1.24, 116.85
    zoom_val = 11

    if single_point_ctx is not None:
        center_lat = (float(single_point_ctx['tx_lat']) + float(single_point_ctx['rx_lat'])) / 2.0
        center_lon = (float(single_point_ctx['tx_lon']) + float(single_point_ctx['rx_lon'])) / 2.0
        zoom_val = 12
    elif df_grid_result is not None and len(df_grid_result) > 0:
        center_lat = float(df_grid_result['center_lat'].mean())
        center_lon = float(df_grid_result['center_lon'].mean())
    elif raster_overlay_data is not None and 'bounds' in raster_overlay_data:
        center_lat = (float(raster_overlay_data['bounds'][0][0]) + float(raster_overlay_data['bounds'][1][0])) / 2.0
        center_lon = (float(raster_overlay_data['bounds'][0][1]) + float(raster_overlay_data['bounds'][1][1])) / 2.0
    elif df_tx_summary is not None and len(df_tx_summary) > 0:
        if selected_tx_name and selected_tx_name != "Semua Pemancar" and selected_tx_name in df_tx_summary['STN_NAME'].values:
            row_sel = df_tx_summary[df_tx_summary['STN_NAME'] == selected_tx_name].iloc[0]
            center_lat = float(row_sel['SID_LAT'])
            center_lon = float(row_sel['SID_LONG'])
        else:
            center_lat = float(df_tx_summary['SID_LAT'].mean())
            center_lon = float(df_tx_summary['SID_LONG'].mean())
    elif df_matched_dt is not None and len(df_matched_dt) > 0:
        center_lat = float(df_matched_dt['Latitude'].mean())
        center_lon = float(df_matched_dt['Longitude'].mean())
    elif service_area_polygon is not None and 'center' in service_area_polygon:
        center_lat = float(service_area_polygon['center'][0])
        center_lon = float(service_area_polygon['center'][1])

    # 2. Initialize Folium Map
    if basemap_choice == "street":
        init_tile = 'openstreetmap'
    else:
        init_tile = None # Will add Esri as first layer

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_val,
        tiles=init_tile,
        control_scale=True,
        scrollWheelZoom=True
    )
    m.add_child(LeafletInteractiveZoomHelper())

    # Add standard high-quality base layers (Satellite & Street)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='🛰️ Esri Satellite (Citra Satelit)',
        show=(basemap_choice != "street")
    ).add_to(m)

    folium.TileLayer(
        'openstreetmap', 
        name='🗺️ OpenStreetMap (Standar Jalan)', 
        show=(basemap_choice == "street")
    ).add_to(m)

    # 2b. Layer: Official DVB-T2 Service Area Boundary (KML Polygon)
    if service_area_polygon is not None and 'polygon' in service_area_polygon and len(service_area_polygon['polygon']) > 0:
        sa_title = service_area_polygon.get('raw_name', 'Wilayah Layanan')
        fg_wl = folium.FeatureGroup(name=f"🌐 Batas Wilayah Layanan ({sa_title})", show=True)
        folium.Polygon(
            locations=service_area_polygon['polygon'],
            color='#0284c7',
            weight=2.8,
            opacity=0.95,
            fill=True,
            fill_color='#0284c7',
            fill_opacity=0.04,
            tooltip=None,
            popup=None,
            class_name='kml-service-area-poly'
        ).add_to(fg_wl)
        fg_wl.add_to(m)

    # 2c. Layer: Official DVB-T2 Test Points (KML Balloon Pin Markers)
    if service_area_testpoints is not None and len(service_area_testpoints) > 0:
        fg_tp = folium.FeatureGroup(name="📍 Titik Uji Evaluasi (Test Points KML)", show=True)
        eval_lookup = {}
        if test_point_evaluations is not None and len(test_point_evaluations) > 0:
            for _, r in test_point_evaluations.iterrows():
                eval_lookup[str(r['Nama Titik Uji'])] = r.to_dict()

        for tp in service_area_testpoints:
            t_lat = float(tp['lat'])
            t_lon = float(tp['lon'])
            t_name = str(tp['name'])
            t_fs = float(tp.get('ref_fs_dbuv', 54.0))

            ev = eval_lookup.get(t_name)
            if ev is not None:
                status_code = ev.get('status_code', 'PASS')
                st_color = 'green' if status_code == 'PASS' else ('orange' if status_code == 'TOLERABLE' else 'red')
                st_icon = 'check' if status_code == 'PASS' else ('exclamation' if status_code == 'TOLERABLE' else 'times')
                pred_fs = float(ev.get('Prediksi Kuat Medan AI (dBµV/m)', ev.get('Prediksi AI (dBµV/m)', 0.0)))
                margin_db = float(ev.get('Margin thd Batas Maks (dB)', ev.get('Margin Kepatuhan (dB)', 0.0)))
                status_txt = str(ev.get('Status Koordinasi', ev.get('Status Kepatuhan', '')))
                dist_km = float(ev.get('Jarak ke TX (km)', 0.0))
                clr_val = ev.get('Clearance Medan (m)')
                is_cut_nlos = bool(ev.get('is_cut_by_nlos', False))
                is_cut_horizon = bool(ev.get('is_cut_by_horizon', False))
                is_est = bool(ev.get('is_estimated', False) or is_cut_nlos or is_cut_horizon or pred_fs < 45.2)
                est_suffix = " (Est.)" if is_est else ""
                status_color = ev.get('color', '#10b981')

                margin_note = "Aman (Di bawah batas maksimum)" if margin_db <= 0 else ("Dalam batas toleransi +3 dB" if margin_db <= 3.0 else "Melanggar (Potensi interferensi)")

                clr_line = f"<b>Clearance Medan:</b> {clr_val:+.1f} m {'(Terhalang Topografi / NLoS)' if (clr_val is not None and clr_val < 0) else '(Line-of-Sight)'}<br/>" if clr_val is not None else ""

                popup_tp = f"""
                <div style="font-family: sans-serif; font-size: 13px; min-width: 250px; line-height: 1.5;">
                    <h4 style="margin: 0 0 5px 0; color: #0284c7; font-size: 14px; font-weight: bold;">📍 {t_name}</h4>
                    <div style="border-bottom: 1px solid #cbd5e1; margin-bottom: 6px;"></div>
                    <b>Status Koordinasi:</b> <span style="font-weight:bold; color:{status_color};">{status_txt}</span><br/>
                    <b>Batas Maksimum Acuan:</b> {t_fs:.1f} dBµV/m<br/>
                    <b>Prediksi Kuat Medan AI:</b> <span style="font-size:14px; font-weight:bold; color:{status_color};">{pred_fs:.1f} dBµV/m{est_suffix}</span><br/>
                    <b>Margin thd Batas Maks:</b> <span style="font-weight:bold; color:{status_color};">{margin_db:+.1f} dB</span> <small>({margin_note})</small><br/>
                    {clr_line}
                    <b>Jarak ke TX:</b> {dist_km:.2f} km<br/>
                    <b>Koordinat:</b> {t_lat:.6f}, {t_lon:.6f}
                    <div style="margin-top: 6px; padding: 4px 6px; background: #f8fafc; border-left: 3px solid {status_color}; font-size: 11px; color: #475569;">
                        🛡️ <i>Titik batas koordinasi untuk mencegah interferensi lintas wilayah siaran (Permenkominfo No. 6/2019).</i>
                    </div>
                </div>
                """
                # Dark glassmorphism tooltip matching probeTooltip format
                if status_code == 'PASS':
                    dot_color = '#10b981' # Emerald Green
                    if is_cut_nlos:
                        badge_label = 'AMAN (NLOS CUTOFF)'
                    elif is_cut_horizon:
                        badge_label = 'AMAN (HORIZON)'
                    elif is_est:
                        badge_label = 'AMAN (BLANK SPOT)'
                    else:
                        badge_label = 'AMAN'
                    margin_color = '#34d399'
                elif status_code == 'TOLERABLE':
                    dot_color = '#f59e0b' # Amber
                    badge_label = 'TOLERANSI (+3 dB)'
                    margin_color = '#fbbf24'
                else:
                    dot_color = '#ef4444' # Red
                    badge_label = 'MELANGGAR'
                    margin_color = '#f87171'

                clr_str = f" | <span style=\"color:#38bdf8;\">Clr={clr_val:+.1f}m</span>" if clr_val is not None else ""

                tooltip_tp_html = f"""<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">
                    <span style="color:{dot_color}; font-weight: 800;">● <b>{t_name}</b> ({badge_label})</span>: 
                    <span style="font-weight: 700; color: #ffffff;">Pred {pred_fs:.1f} dBµV/m{est_suffix}</span> | 
                    <span style="color:#cbd5e1;">Batas Maks {t_fs:.1f} dBµV/m</span> | 
                    <span style="color:{margin_color}; font-weight: 700;">Margin {margin_db:+.1f} dB</span> | 
                    <span style="color:#cbd5e1;">d={dist_km:.2f} km</span>{clr_str}
                </div>"""
                icon_obj = folium.Icon(color=st_color, icon=st_icon, prefix='fa')
            else:
                popup_tp = f"""
                <div style="font-family: sans-serif; font-size: 13px; min-width: 220px; line-height: 1.5;">
                    <h4 style="margin: 0 0 5px 0; color: #0284c7; font-size: 13px; font-weight: bold;">📍 {t_name}</h4>
                    <div style="border-bottom: 1px solid #cbd5e1; margin-bottom: 6px;"></div>
                    <b>Batas Maksimum Acuan:</b> {t_fs:.1f} dBµV/m<br/>
                    <b>Koordinat:</b> {t_lat:.6f}, {t_lon:.6f}<br/>
                    <div style="margin-top: 6px; padding: 4px 6px; background: #f8fafc; border-left: 3px solid #0284c7; font-size: 11px; color: #475569;">
                        🛡️ <i>Titik batas koordinasi pencegahan interferensi ke wilayah siaran tetangga.</i>
                    </div>
                </div>
                """
                tooltip_tp_html = f"""<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">
                    <span style="color:#38bdf8; font-weight: 800;">● <b>{t_name}</b></span>: 
                    <span style="color:#cbd5e1;">Batas Maks {t_fs:.1f} dBµV/m</span> | 
                    <span style="color:#94a3b8;">Koordinat {t_lat:.4f}, {t_lon:.4f}</span>
                </div>"""
                icon_obj = folium.Icon(color='blue', icon='flag', prefix='fa')

            folium.Marker(
                location=[t_lat, t_lon],
                popup=folium.Popup(popup_tp, max_width=290),
                tooltip=folium.Tooltip(
                    tooltip_tp_html,
                    class_name="cloudrf-grid-tooltip",
                    sticky=False,
                    direction="top",
                    offset=[0, -10],
                    opacity=0.98
                ),
                icon=icon_obj
            ).add_to(fg_tp)
        fg_tp.add_to(m)

    # 3a. Layer: Continuous Telecom Raster Heatmap (CloudRF Style)
    if show_coverage_grid and raster_overlay_data is not None and 'image_b64' in raster_overlay_data:
        folium.raster_layers.ImageOverlay(
            image=raster_overlay_data['image_b64'],
            bounds=raster_overlay_data['bounds'],
            opacity=raster_overlay_data.get('opacity', 0.75),
            name="🎨 Peta Cakupan Sinyal (CloudRF Continuous Heatmap)",
            interactive=False,
            cross_origin=False
        ).add_to(m)

        if 'fs_grid_b64' in raster_overlay_data:
            m.add_child(RasterCoverageInspector(raster_overlay_data))

    # 3b. Layer: Coverage Grid Polygons (100m / 50m / 10m)
    if show_coverage_grid and df_grid_result is not None and len(df_grid_result) > 0 and 'lat_min' in df_grid_result.columns:
        res_m = float(df_grid_result.iloc[0].get('resolution_m', 100.0)) if 'resolution_m' in df_grid_result.columns else 100.0
        fg_grid = folium.FeatureGroup(name=f"🟩 Poligon Cakupan Sinyal (Kotak {res_m:.0f}m x {res_m:.0f}m)", show=True)
        
        covered_rows = df_grid_result[df_grid_result.get('is_covered', True)]
        if len(covered_rows) > 9000:
            covered_rows = covered_rows.head(9000)

        border_w = 0.15 if res_m <= 10.0 else (0.25 if res_m <= 50.0 else 0.4)

        for _, row in covered_rows.iterrows():
            lat_min = float(row['lat_min'])
            lat_max = float(row['lat_max'])
            lon_min = float(row['lon_min'])
            lon_max = float(row['lon_max'])
            fs = float(row['median_field_strength_dbuv_m'])
            cat = str(row['category'])
            c = str(row['color'])
            opacity = float(row.get('fill_opacity', 0.65))
            d_km = float(row['distance_km'])
            clr = row.get('clearance_m')
            clr_str = f" | Clr={clr:+.1f}m" if clr is not None and not pd.isna(clr) else ""

            folium.Rectangle(
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                color=c,
                weight=border_w,
                fill=True,
                fill_color=c,
                fill_opacity=opacity,
                tooltip=f"<b>{cat}</b>: {fs:.1f} dBµV/m | d={d_km:.2f} km{clr_str}"
            ).add_to(fg_grid)
        fg_grid.add_to(m)

    # 4. Layer: Drive-Test Points
    if show_drive_test and df_matched_dt is not None and len(df_matched_dt) > 0:
        fg_dt = folium.FeatureGroup(name="🚗 Titik Hasil Ukur Lapangan (Drive Test)", show=True)
        df_dt_plot = df_matched_dt
        if selected_tx_name and selected_tx_name != "Semua Pemancar" and 'STN_NAME' in df_matched_dt.columns:
            df_dt_plot = df_matched_dt[df_matched_dt['STN_NAME'] == selected_tx_name]
        
        # Subsample for rendering speed if dense
        if len(df_dt_plot) > 3500:
            df_dt_plot = df_dt_plot.sample(n=3500, random_state=42)

        for _, row in df_dt_plot.iterrows():
            r_lat = float(row['Latitude'])
            r_lon = float(row['Longitude'])
            fs = float(row.get('field_strength_dbuv_m', row.get('Level_dBuV_m', 0.0)))
            d_km = float(row.get('distance_km', 0.0))
            stn = str(row.get('STN_NAME', 'Stasiun'))

            if fs >= 65.0:
                c = '#ef4444' # Merah (Sangat Kuat)
            elif fs >= 54.0:
                c = '#f97316' # Oranye (Kuat)
            elif fs >= 45.2:
                c = '#3b82f6' # Biru (Sedang)
            else:
                c = '#94a3b8' # Abu-abu netral (Blank Spot)

            folium.CircleMarker(
                location=[r_lat, r_lon],
                radius=3.0,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.85,
                weight=0,
                tooltip=f"{stn} | Level: {fs:.1f} dBµV/m | d={d_km:.2f} km"
            ).add_to(fg_dt)
        fg_dt.add_to(m)

    # 5. Layer: Transmitters (Tower Pins)
    if show_transmitters and df_tx_summary is not None and len(df_tx_summary) > 0:
        fg_tx = folium.FeatureGroup(name="📡 Menara Pemancar DVB-T2", show=True)
        for _, row in df_tx_summary.iterrows():
            stn = str(row['STN_NAME'])
            lat = float(row['SID_LAT'])
            lon = float(row['SID_LONG'])
            f_mhz = float(row['FREQ_MHZ'])
            h_ant = float(row.get('H_ANT_masl', row.get('H_ANT', 50.0)))
            eirp = float(row.get('EIRP_dBm', 0.0))
            is_active_stn = bool(selected_tx_name and selected_tx_name == stn)

            # CloudRF style orange/blue pin
            pin_color = 'red' if is_active_stn else 'orange'
            icon_name = 'star' if is_active_stn else 'broadcast-tower'

            popup_tx = f"""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 13px; min-width: 220px; line-height: 1.5;">
                <h4 style="margin: 0 0 6px 0; color: #1e3a8a; font-size: 14px; font-weight: bold;">📡 {stn}</h4>
                <div style="border-bottom: 1px solid #cbd5e1; margin-bottom: 8px;"></div>
                <b>Frekuensi:</b> {f_mhz:.2f} MHz (UHF)<br/>
                <b>Koordinat:</b> {lat:.6f}, {lon:.6f}<br/>
                <b>Tinggi Antena:</b> {h_ant:.1f} m AMSL<br/>
                <b>EIRP Terhitung:</b> <span style="color:#b91c1c; font-weight:bold;">{eirp:.2f} dBm</span><br/>
            </div>
            """

            tooltip_tx_html = f"""<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">
                <span style="color:#38bdf8; font-weight:800;">📡 <b>{stn}</b></span>: 
                <span style="font-weight:700; color:#ffffff;">{f_mhz:.1f} MHz</span> | 
                <span style="color:#fde047; font-weight:700;">EIRP {eirp:.1f} dBm</span> | 
                <span style="color:#cbd5e1;">H_ANT {h_ant:.0f}m</span>
            </div>"""
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_tx, max_width=300),
                tooltip=folium.Tooltip(
                    tooltip_tx_html,
                    class_name="cloudrf-grid-tooltip",
                    sticky=False,
                    direction="top",
                    offset=[0, -10],
                    opacity=0.98
                ),
                icon=folium.Icon(color=pin_color, icon=icon_name, prefix='fa')
            ).add_to(fg_tx)

        fg_tx.add_to(m)

    # 6. Layer: Single Point Prediction Ray & Target Marker
    if single_point_ctx is not None:
        fg_sp = folium.FeatureGroup(name="🎯 Titik Penerima (Rx)", show=True)
        rx_c = [float(single_point_ctx['rx_lat']), float(single_point_ctx['rx_lon'])]

        if single_point_res is not None:
            tx_c = [float(single_point_ctx['tx_lat']), float(single_point_ctx['tx_lon'])]
            line_color = str(single_point_res.get('color', '#3b82f6'))
            folium.PolyLine(
                locations=[tx_c, rx_c],
                color=line_color,
                weight=3.5,
                opacity=0.9,
                dash_array='8, 6',
                tooltip=f"Lintasan Propagasi: {single_point_res.get('distance_km', 0.0):.2f} km | {single_point_res.get('predicted_field_strength_dbuv_m', 0.0):.1f} dBµV/m"
            ).add_to(fg_sp)

            # Receiver marker
            rx_fs = single_point_res.get('predicted_field_strength_dbuv_m', 0.0)
            rx_cat = single_point_res.get('category', 'N/A')
            rx_clr = single_point_res.get('clearance_m', 0.0)
            is_sp_est = bool(single_point_res.get('is_estimated', False) or rx_fs < 45.2 or rx_cat == 'Blank Spot')
            sp_est_suffix = " (Est.)" if is_sp_est else ""

            popup_rx = f"""
            <div style="font-family: sans-serif; font-size: 13px; min-width: 200px;">
                <h4 style="margin: 0 0 5px 0; color: #0284c7;">📍 Lokasi Penerima (Rx)</h4>
                <b>Koordinat:</b> {rx_c[0]:.6f}, {rx_c[1]:.6f}<br/>
                <b>Kuat Medan AI:</b> <span style="font-weight:bold; color:{line_color};">{rx_fs:.1f} dBµV/m{sp_est_suffix}</span><br/>
                <b>Kategori:</b> <b>{rx_cat}</b><br/>
                <b>Clearance Topografi:</b> {rx_clr:+.2f} m<br/>
                <b>Jarak ke Tx:</b> {single_point_res.get('distance_km', 0.0):.2f} km
            </div>
            """
            rx_cat_color = '#ef4444' if rx_cat == 'Sangat Kuat' else ('#f97316' if rx_cat == 'Kuat' else ('#3b82f6' if rx_cat == 'Sedang' else '#94a3b8'))
            tooltip_rx_html = f"""<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">
                <span style="color:{rx_cat_color}; font-weight:800;">🎯 <b>Penerima (Rx)</b> ({rx_cat})</span>: 
                <span style="font-weight:700; color:#ffffff;">{rx_fs:.1f} dBµV/m{sp_est_suffix}</span> | 
                <span style="color:#cbd5e1;">d={single_point_res.get('distance_km', 0.0):.2f} km</span>
            </div>"""
            folium.Marker(
                location=rx_c,
                popup=folium.Popup(popup_rx, max_width=280),
                tooltip=folium.Tooltip(
                    tooltip_rx_html,
                    class_name="cloudrf-grid-tooltip",
                    sticky=False,
                    direction="top",
                    offset=[0, -10],
                    opacity=0.98
                ),
                icon=folium.Icon(color='green' if rx_fs >= 45.2 else 'gray', icon='user', prefix='fa')
            ).add_to(fg_sp)
        else:
            tooltip_sel_html = f"""<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 12px; font-weight: 500; line-height: 1.4; color: #f8fafc; white-space: nowrap;">
                <span style="color:#38bdf8; font-weight:800;">📍 <b>Titik Terpilih</b></span>: 
                <span style="color:#cbd5e1;">{rx_c[0]:.6f}, {rx_c[1]:.6f}</span>
            </div>"""
            folium.Marker(
                location=rx_c,
                tooltip=folium.Tooltip(
                    tooltip_sel_html,
                    class_name="cloudrf-grid-tooltip",
                    sticky=False,
                    direction="top",
                    offset=[0, -10],
                    opacity=0.98
                ),
                icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')
            ).add_to(fg_sp)

        fg_sp.add_to(m)

    # 7. CloudRF Style Vertical Colorbar Legend (Narrow & Tall Stacked Boxes in Upper-Right Corner)
    colorbar_html = """
    <div style="
        position: fixed; 
        top: 10px; right: 10px; 
        width: 32px;
        background: transparent !important; 
        border: none !important; 
        z-index: 999999 !important; 
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; 
        text-align: center !important; 
        pointer-events: none; 
    ">
        <div style="
            background: rgba(15, 23, 42, 0.75);
            color: #38bdf8;
            font-weight: 800;
            font-size: 8.5px;
            padding: 4px 1px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            border: 1px solid rgba(0, 0, 0, 0.5);
            border-bottom: none;
            text-shadow: 0 1px 2px rgba(0,0,0,0.9);
            letter-spacing: -0.3px;
            line-height: 1.1;
        ">
            dBµV/m
        </div>
        <div style="
            display: flex;
            flex-direction: column;
            width: 32px;
            border: 1px solid rgba(0, 0, 0, 0.6);
            border-bottom-left-radius: 4px;
            border-bottom-right-radius: 4px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        ">
            <div style="
                background: #ef4444;
                color: #ffffff;
                font-size: 10px;
                font-weight: 800;
                height: 38px;
                width: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                border-bottom: 1px solid rgba(0,0,0,0.25);
            " title="Sangat Kuat (≥65 dBµV/m)">
                ≥65
            </div>
            <div style="
                background: #f97316;
                color: #ffffff;
                font-size: 10px;
                font-weight: 800;
                height: 38px;
                width: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                border-bottom: 1px solid rgba(0,0,0,0.25);
            " title="Kuat (54 – 65 dBµV/m)">
                54
            </div>
            <div style="
                background: #3b82f6;
                color: #ffffff;
                font-size: 9.5px;
                font-weight: 800;
                height: 38px;
                width: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-shadow: 0 1px 2px rgba(0,0,0,0.8);
                border-bottom: 1px solid rgba(0,0,0,0.25);
            " title="Sedang (45.2 – 54 dBµV/m)">
                45.2
            </div>
            <div style="
                background: rgba(15, 23, 42, 0.40);
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
                color: #f1f5f9;
                font-size: 9px;
                font-weight: 800;
                height: 38px;
                width: 32px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-shadow: 0 1px 3px rgba(0,0,0,0.95);
            " title="Blank Spot (<45.2 dBµV/m)">
                &lt;45
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(colorbar_html))

    folium.LayerControl(position='topleft', collapsed=True).add_to(m)

    return m


def create_input_data_map(
    df_tx_summary: pd.DataFrame,
    df_matched_dt: pd.DataFrame,
    selected_stn: Optional[str] = None
) -> folium.Map:
    """
    Constructs the Tab 1 Data Ingestion Map:
    - Transmitter Tower Markers with technical popups
    - Raw Drive Test route points colored by measured Level (dBuV/m)
    - Full street and geographic names from OpenStreetMap / CartoDB
    """
    # Calculate geographical center
    center_lat = float(df_tx_summary['SID_LAT'].mean())
    center_lon = float(df_tx_summary['SID_LONG'].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None,
        control_scale=True,
        scrollWheelZoom=True
    )
    m.add_child(LeafletInteractiveZoomHelper())

    # 1. Base Tile Layers (Esri Satellite & OpenStreetMap)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='🛰️ Esri Satellite (Citra Satelit)'
    ).add_to(m)
    folium.TileLayer('openstreetmap', name='🗺️ OpenStreetMap (Standar Jalan)').add_to(m)

    # 2. Transmitter Sites FeatureGroup
    fg_tx = folium.FeatureGroup(name="📡 Menara Pemancar DVB-T2", show=True)

    for _, row in df_tx_summary.iterrows():
        lat = float(row['SID_LAT'])
        lon = float(row['SID_LONG'])
        stn = str(row['STN_NAME'])
        f_mhz = float(row['FREQ_MHZ'])
        h_ant = float(row['H_ANT_masl'])
        eirp = float(row['EIRP_dBm'])
        e_max = float(row.get('E_max_dBuV_m', 0.0))
        d_max = float(row.get('d_at_Emax_km', 0.0))

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 220px;">
            <h4 style="margin: 0 0 5px 0; color: #1f77b4;">📡 {stn}</h4>
            <hr style="margin: 4px 0 8px 0; border: 0.5px solid #ccc;"/>
            <b>Frekuensi:</b> {f_mhz:.2f} MHz (UHF)<br/>
            <b>Koordinat:</b> {lat:.6f}, {lon:.6f}<br/>
            <b>Tinggi Antena:</b> {h_ant:.1f} m AMSL<br/>
            <b>EIRP Terhitung:</b> <span style="color:#d9534f; font-weight:bold;">{eirp:.2f} dBm</span><br/>
            <b>Kuat Medan Maks:</b> {e_max:.1f} dBµV/m (pada jarak {d_max:.2f} km)<br/>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{stn} ({f_mhz:.2f} MHz | EIRP: {eirp:.1f} dBm)",
            icon=folium.Icon(color='darkblue', icon='broadcast-tower', prefix='fa')
        ).add_to(fg_tx)

    fg_tx.add_to(m)

    # 3. Drive Test Measurements FeatureGroup
    fg_dt = folium.FeatureGroup(name="🚗 Titik Ukur Drive Test (Ground Truth)", show=True)

    df_dt_plot = df_matched_dt
    if selected_stn and selected_stn != "Semua Pemancar":
        df_dt_plot = df_matched_dt[df_matched_dt['STN_NAME'] == selected_stn]

    # Downsample points slightly for browser responsiveness if massive
    if len(df_dt_plot) > 4000:
        df_dt_plot = df_dt_plot.sample(n=4000, random_state=42)

    for _, row in df_dt_plot.iterrows():
        r_lat = float(row['Latitude'])
        r_lon = float(row['Longitude'])
        fs = float(row['field_strength_dbuv_m'])
        d_km = float(row['distance_km'])
        stn = str(row['STN_NAME'])

        # Color mapping (Merah, Oranye, Biru, Abu-abu)
        if fs >= 65.0:
            c = '#ef4444' # Merah (Sangat Kuat)
        elif fs >= 54.0:
            c = '#f97316' # Oranye (Kuat)
        elif fs >= 45.2:
            c = '#3b82f6' # Biru (Sedang)
        else:
            c = '#94a3b8' # Abu-abu netral (Blank Spot)

        folium.CircleMarker(
            location=[r_lat, r_lon],
            radius=3.0,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.8,
            weight=0,
            tooltip=f"{stn} | Level: {fs:.1f} dBµV/m | d={d_km:.2f} km"
        ).add_to(fg_dt)

    fg_dt.add_to(m)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    return m


def create_polygon_coverage_map(
    df_grid_result: pd.DataFrame,
    tx_params: Dict[str, Any],
    radius_km: float = 30.0,
    show_drive_test: bool = False,
    df_drivetest: Optional[pd.DataFrame] = None
) -> folium.Map:
    """
    Constructs the Tab 3 Coverage Area Map:
    - Selected Transmitter Tower Marker
    - Radius circle limit (dashed line)
    - 100m x 100m Rectangle Polygons colored by 4 DVB-T2 Tiers (Blank spot transparent)
    - Interactive Legend and Full Street Labels
    """
    tx_lat = float(tx_params['SID_LAT'])
    tx_lon = float(tx_params['SID_LONG'])
    stn_name = str(tx_params['STN_NAME'])
    freq_mhz = float(tx_params['FREQ_MHZ'])
    eirp = float(tx_params['EIRP_dBm'])
    h_ant = float(tx_params['H_ANT_masl'])

    m = folium.Map(
        location=[tx_lat, tx_lon],
        zoom_start=11,
        tiles=None,
        control_scale=True,
        scrollWheelZoom=True
    )
    m.add_child(LeafletInteractiveZoomHelper())

    # Base layers (Esri Satellite & OpenStreetMap)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='🛰️ Esri Satellite (Citra Satelit)'
    ).add_to(m)
    folium.TileLayer('openstreetmap', name='🗺️ OpenStreetMap (Standar Jalan)').add_to(m)

    # 1. Transmitter Marker
    fg_tx = folium.FeatureGroup(name=f"📡 Pemancar: {stn_name}", show=True)

    popup_tx = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px; min-width: 200px;">
        <h4 style="margin: 0 0 5px 0; color: #1f77b4;">📡 {stn_name}</h4>
        <b>Frekuensi:</b> {freq_mhz:.2f} MHz<br/>
        <b>Tinggi Antena:</b> {h_ant:.1f} m AMSL<br/>
        <b>EIRP:</b> {eirp:.2f} dBm<br/>
        <b>Radius Simulasi:</b> {radius_km} km<br/>
    </div>
    """

    folium.Marker(
        location=[tx_lat, tx_lon],
        popup=folium.Popup(popup_tx, max_width=300),
        tooltip=f"{stn_name} ({eirp:.1f} dBm)",
        icon=folium.Icon(color='red', icon='broadcast-tower', prefix='fa')
    ).add_to(fg_tx)

    # Draw Radius Circle boundary
    folium.Circle(
        location=[tx_lat, tx_lon],
        radius=radius_km * 1000.0,
        color='#2980b9',
        weight=2,
        fill=False,
        dash_array='6, 6',
        tooltip=f"Batas Radius {radius_km} km"
    ).add_to(fg_tx)

    fg_tx.add_to(m)

    # 2. Rectangle Polygons Grid Layer (100m / 50m / 10m)
    res_val = float(df_grid_result.iloc[0].get('resolution_m', 100.0)) if 'resolution_m' in df_grid_result.columns else 100.0
    fg_coverage = folium.FeatureGroup(name=f"🟩 Cakupan Sinyal DVB-T2 (Kotak Poligon {res_val:.0f}m x {res_val:.0f}m)", show=True)
    border_w = 0.15 if res_val <= 10.0 else (0.25 if res_val <= 50.0 else 0.4)

    covered_rows = df_grid_result[df_grid_result['is_covered']]
    if len(covered_rows) > 9000:
        covered_rows = covered_rows.head(9000)

    for _, row in covered_rows.iterrows():
        lat_min = float(row['lat_min'])
        lat_max = float(row['lat_max'])
        lon_min = float(row['lon_min'])
        lon_max = float(row['lon_max'])
        fs = float(row['median_field_strength_dbuv_m'])
        cat = str(row['category'])
        c = str(row['color'])
        opacity = float(row['fill_opacity'])
        d_km = float(row['distance_km'])

        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        clr = row.get('clearance_m')
        clr_str = f" | Clr={clr:+.1f}m" if clr is not None and not pd.isna(clr) else ""
        tip = f"<b>{cat}</b>: {fs:.1f} dBµV/m | d={d_km:.2f} km{clr_str}"

        folium.Rectangle(
            bounds=bounds,
            color=c,
            weight=border_w,
            fill=True,
            fill_color=c,
            fill_opacity=opacity,
            tooltip=tip
        ).add_to(fg_coverage)

    fg_coverage.add_to(m)

    # 3. Optional Ground Truth Drive Test Layer
    if show_drive_test and df_drivetest is not None and len(df_drivetest) > 0:
        fg_dt = folium.FeatureGroup(name="Titik Drive Test Lapangan Asli", show=False)
        dt_sample = df_drivetest.sample(n=min(len(df_drivetest), 2500), random_state=42)
        for _, row in dt_sample.iterrows():
            r_lat = float(row['Latitude'])
            r_lon = float(row['Longitude'])
            fs = float(row['field_strength_dbuv_m'])
            folium.CircleMarker(
                location=[r_lat, r_lon],
                radius=2.0,
                color='#34495e',
                fill=True,
                fill_color='#34495e',
                fill_opacity=0.8,
                weight=0,
                tooltip=f"Real Field Strength: {fs:.1f} dBµV/m"
            ).add_to(fg_dt)
        fg_dt.add_to(m)

    # 4. Floating Legend Box (Semi-Transparent Glassmorphism styling)
    legend_html = f"""
    <div style="
        position: fixed; 
        bottom: 30px; left: 30px; width: 285px; 
        background: rgba(15, 23, 42, 0.70) !important; 
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        color: #f8fafc !important; 
        border: 1px solid rgba(255, 255, 255, 0.22) !important; 
        z-index: 999999 !important; 
        font-size: 12px !important;
        padding: 12px 14px !important; 
        border-radius: 8px !important; 
        box-shadow: 0 8px 30px rgba(0,0,0,0.40) !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        line-height: 1.5 !important;
    ">
        <div style="color: #38bdf8 !important; font-weight: bold; font-size: 13px; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.18); padding-bottom: 4px;">
            📡 Kualitas Sinyal DVB-T2 (Band IV/V)
        </div>
        <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="background: #ef4444; width: 16px; height: 16px; display: inline-block; border-radius: 3px; border: 1px solid rgba(255,255,255,0.3); margin-right: 8px; flex-shrink: 0;"></span>
            <span style="color: #f1f5f9 !important; font-size: 12px;"><b>Sangat Kuat:</b> &ge; 65.0 dBµV/m (Merah)</span>
        </div>
        <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="background: #f97316; width: 16px; height: 16px; display: inline-block; border-radius: 3px; border: 1px solid rgba(255,255,255,0.3); margin-right: 8px; flex-shrink: 0;"></span>
            <span style="color: #f1f5f9 !important; font-size: 12px;"><b>Kuat:</b> 54.0 – 65.0 dBµV/m (Oranye)</span>
        </div>
        <div style="margin-bottom: 6px; display: flex; align-items: center;">
            <span style="background: #3b82f6; width: 16px; height: 16px; display: inline-block; border-radius: 3px; border: 1px solid rgba(255,255,255,0.3); margin-right: 8px; flex-shrink: 0;"></span>
            <span style="color: #f1f5f9 !important; font-size: 12px;"><b>Sedang:</b> 45.2 – 54.0 dBµV/m (Biru)</span>
        </div>
        <div style="margin-bottom: 8px; display: flex; align-items: center;">
            <span style="background: rgba(255,255,255,0.05); border: 1.5px dashed #94a3b8; width: 16px; height: 16px; display: inline-block; border-radius: 3px; margin-right: 8px; flex-shrink: 0;"></span>
            <span style="color: #94a3b8 !important; font-size: 12px;"><b>Blank Spot:</b> &lt; 45.2 dBµV/m (Bening)</span>
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.18); padding-top: 6px; color: #cbd5e1 !important; font-size: 11px; line-height: 1.4;">
            <span style="color: #cbd5e1 !important;">📡 Pemancar: <b style="color: #f8fafc !important;">{stn_name}</b></span><br/>
            <span style="color: #cbd5e1 !important;">🟦 Mode: <b style="color: #f8fafc !important;">Telecom Heatmap AI Coverage</b></span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    return m


def plot_parity_scatter(y_true: np.ndarray, y_pred: np.ndarray, r2: float, rmse: float) -> go.Figure:
    """Creates Parity Plot (Actual vs Predicted) with ideal 1:1 diagonal."""
    min_val = min(float(np.min(y_true)), float(np.min(y_pred))) - 2
    max_val = max(float(np.max(y_true)), float(np.max(y_pred))) + 2

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(
            color=y_pred,
            colorscale='Viridis',
            size=6,
            opacity=0.65,
            showscale=True,
            colorbar=dict(
                title=dict(text='Pred (dBµV/m)', side='top'),
                thickness=16,
                len=0.88,
                y=0.5,
                yanchor='middle'
            )
        ),
        name='Data Sampel Uji'
    ))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Garis Ideal (1:1)'
    ))

    fig.update_layout(
        title=f"Evaluasi Model: Aktual vs Prediksi (R² = {r2:.4f}, Test RMSE = {rmse:.2f} dBµV/m)",
        xaxis_title="Kuat Medan Aktual / Lapangan (dBµV/m)",
        yaxis_title="Kuat Medan Prediksi XGBoost (dBµV/m)",
        template='plotly_white',
        height=450,
        margin=dict(l=50, r=40, t=50, b=50),
        legend=dict(
            x=0.03,
            y=0.97,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.88)',
            bordercolor='#cbd5e1',
            borderwidth=1,
            font=dict(size=12)
        )
    )
    return fig


def plot_feature_importance(feature_importances: Dict[str, float]) -> go.Figure:
    """Plots horizontal bar chart of XGBoost feature importances."""
    df_imp = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
    df_imp = df_imp.sort_values(by='Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker=dict(
            color=df_imp['Importance'],
            colorscale='Teal',
            showscale=False
        )
    ))

    fig.update_layout(
        title="Bobot Kepentingan Fitur Fisik Model (Gain Importance)",
        xaxis_title="Relative Importance (Gain)",
        yaxis_title="Fitur Input Fisik",
        template='plotly_white',
        height=380,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
