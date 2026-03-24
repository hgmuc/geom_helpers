ELEVATION MAP
=============

# Schritte (aktueller Stand)

1. Aus den Natural Earth GeoJSON Dateien die relevanten Daten (BBOX für Europa) herausfiltern und in neue GeoJSON Dateien speichern
	- Prozess implementiert in **osmium/NaturalEarth2VectorTiles.ipynb** Notebook
	- Rohdaten in **osmium/natural_earth/raw**  (Quelle: https://github.com/nvkelso/natural-earth-vector/tree/master/geojson)
	- Output in **osmium/natural_earth**

2. Die gefilterten GeoJSON Dateien einzeln einlesen und einzelne Layer für Vector Tiles erzeugen (z.B. Länder, Flüsse, Straßen, Ländernamen, ...)
	- Prozess implementiert in **osmium/NaturalEarth2VectorTiles.ipynb** Notebook
	- Outputs:
		- Je Layer in **geojson2vt.TileIndex**
		- Aus dem geojson2vt.TileIndex werden bis einschließlich Zoom Level 9 (Param INDEX_MAX_ZOOM) z/x/y.MVT Vector Tile Dateien, direkt gespeichert in Zielordner ("C:/05_Python/tiles1")
		- Für höhere Zoom-Level wird die geojson2vt.TileIndex gepickled und in **osmium/natural_earth/tiles_index** gespeichert.
	- **Hinweis**: Jupyter Notebooks können auch direkt in Visual Studio Code erstellt, bearbeitet und ausgeführt werden.
		- Anleitung: https://code.visualstudio.com/docs/datascience/jupyter-notebooks
		
3. Anzeige mit Web-/Tileserver:
	- Tileserver starten:
		- cd nach **C:\Tools\Anaconda3\envs\osmox**
		- starten mit **python c:/05_Python/tiles/tileserver2.py**
		- ACHTUNG: die meisten Daten (index.html, PNG und MVT liegen zurzeit noch in **c:/05_Python/tiles1** und werden auch dorthin geschrieben. **Die Ordner sollten zusammengefasst werden!!**
	- Der Tileserver liefert sowohl die PNG Raster Tiles als auch die MVT Vector Tiles aus.
	- Ist ein Tile noch nicht vorhanden, wird es zur Laufzeit erzeugt. **Online-Verbindung zum Laden von Elevation Data Dateien notwendig!**
	- Raster Tiles:
		- Für nicht vorhandene Tiles, werden in Abhängigkeit vom Zoom-Level folgende Quellen als Grundlage zum Erzeugen der Heatmap Grafiken verwendet:
			- Mapzen Elevation XYZ Tiles von Amazon S3 (bis Zoom-Level 7)
			- SRTM Dateien (soweit bereits lokal auf dem Rechner vorhanden)
			- Copernicus DEM Dateien von Amazon S3 heruntergeladen (mit 90 bzw 30 Meter Auflösung ab Zoom-Level 10 bzw. 12)
		- Viele der notwendigen Dateien sind bereits auf dem lokalen Rechner und zwar in folgenden Verzeichnissen:
			- Mapzen:     **C:\05_Python\awstiles\terrarium**
			- SRTM:       **C:\SRTM2**
			- Copernicus: **C:\05_Python\awstiles\copernicus\90  bzw.  30**  
		- die unterschiedlichen Array-Shapes der verschiedenen Datenquellen (betrifft nur SRTM und Copernicus) werden in Übereinstimmung gebracht.
		- Es wird zunächst eine Heatmap (mit Seaborn) erzeugt, die anschließend auf mit imageio auf die Tile-Größe 256x256 gebracht wird.
		- Die erzeugten Raster Tiles werden im File System gespeichert, so dass sie der Webserver bei der nächsten Anfrage direkt ausliefern kann
		- **Hinweis:** Bei den Copernicus-Dateien handelt es sich um COG (Cloud-Optimized Geotiffs) Dateien. Diese unterstützen das interessante Feature HTTP Range Request, d.h. es muss nicht die gesamte Datei heruntergeladen werden, sondern es können explizit einzelne Bereiche abgefragt werden. (für dieses Projekt aber nicht interessant).
	- Vector Tiles:
		- Tileserver lädt beim Hochfahren die Tile Index-Dateien für die einzelnen Layer aus den in Schritt 2 gepickelten Dateien.
			- Welche Layer geladen werden hängt vom **layers_def** Dictionary ab, das in **vector_tiles.py** definiert ist und **mit dem gleichnamigen Dictionary im Notebook übereinstimmen sollte!**
		- Kommt ein Request für Zoom-Level 10+ (Param INDEX_MAX_ZOOM), wird die MVT Response aus den Tile Index-Dateien mit den vorab generierten Shapes on-the-fly zusammengebaut.
	- **INDEX.html**
		- Im Verzeichnis **c:/05_Python/tiles1** liegt auch die index.html Datei, die eine Browser-Fenster füllende Map darstellt.
			- Fullscreen Plugin (JS) liegt vor, wird aber hier nicht mehr genutzt.
		- Die Datei enthält praktisch den gesamten Quellcode.
		- Die Datei wird mit http://localhost:8080/ aufgerufen.
		- Alternativ gibt es im **Hugo** Verzeichnis noch die **Site "elevationmap"**. Der Code in der dortigen map.html Shortcode Datei ist aber nicht ganz aktuell.


4. Final Thoughts
	- Die NaturalEarth Daten sind für eine Übersicht OK, aber für höhere Zoom-Level nicht detailliert genug.
	- Bei der aktuellen Version könnten noch die Ozeane und Bundesländer-Grenzen ergänzt werden. (Daten liegen schon lokal vor).
	- Für genauere Daten müssten aus dem OSM Dateien GeoJSON Dateien erzeugt werden (z.B. per Layer oder per Layer und BBox), die dann zu MVT Tiles konvertiert werden können.
	
	
