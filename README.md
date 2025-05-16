
# Objektumfelismerés Gazebo szimulációban


[//]: # (Image References)

[image1]: ./assets/running.jpg "Project in running"

## Tartalomjegyzék

- [A projektben részt vevő személyek](#a_projektben_részt_vevő_személyek)
- [Leírás](#leírás)
- [Követelmények](#követelmények)
- [A kód működése](#a-kód-működése)
  - [Fő osztály: ObjectDetectionNode](#fő-osztály-objectdetectionnode)
    - [__init__() függvény](#init-függvény)
    - [display_image(self) függvény](#display_imageself-függvény)
    - [Egyéb függvények](#egyéb-függvények)
- [Turtlebot3 beállítás](#turtlebot3-beállítás)
  - [Kamera nézet beállítása](#kamera-nézet-beállítása)
- [Futtatás](#futtatás)
  - [Szimuláció indítása](#szimuláció-indítása)
  - [Robot mozgatása](#robot-mozgatása)
  - [Objektum detektálás indítása](#objektum-detektálás-indítása)
- [Összefoglalás](#összefoglalás)

## A projektben részt vevő személyek

Prónai Sára Tímea - FCNR1H
Gyaraki Dániel - S4HGT8
Kazai Kornél  - BW5LTQ
Kőrösi Katinka Flóra - G3I5MM
Ördög Viktor Roland - G6RQSH

## Leírás

Építettünk egy Gazebo világot, amelyben teszteljük az objektumfelismerést. Az alapvető Gazebo modelleket kiegészítettük továbbiakkal, hogy változatosabb legyen a környezet.

Az objektumdetektálást egy Python node végzi, amely az `object_detection_py` mappában található.

## Követelmények

A futtatáshoz az alábbi Python könyvtárak telepítése szükséges (a szokásosakon felül):

```bash
pip install ultralytics opencv-python
pip install torch
pip install "numpy<2.0"
```

## A kód működése

A kód nagyjából 95 soros. Az elején importáljuk a szükséges könyvtárakat: `torch`, `ultralytics`, `sensor_msgs.msg`-ből a `CompressedImage` stb.

### Fő osztály: `ObjectDetectionNode`

A fő logika az `ObjectDetectionNode` osztályban található, amely négy fontosabb függvényt tartalmaz. Az `__init__()` függvény felelős:

- a YOLOv8 modell betöltéséért (`yolov8n.pt`)
- a `CvBridge` inicializálásáért
- a ROS topikra való feliratkozásért (`/image_raw/compressed`)
- a megjelenítő szál indításáért

#### `__init__()` függvény

```python
def __init__(self):
    super().__init__('object_detection_node')

    # YOLOv8 modell betöltése (automatikusan letölti, ha szükséges)
    self.model = YOLO('yolov8n.pt')  # Használható pl. 'yolov8s.pt', 'yolov8m.pt' is

    # CvBridge inicializálása
    self.bridge = CvBridge()

    # Kép előfizetés
    self.subscription = self.create_subscription(
        CompressedImage,
        '/image_raw/compressed',
        self.image_callback,
        10
    )

    # Legfrissebb kép tárolása
    self.latest_frame = None
    self.frame_lock = threading.Lock()

    # Megjelenítő szál indítása
    self.running = True
    self.display_thread = threading.Thread(target=self.display_image)
    self.display_thread.start()
```

### `display_image(self)` függvény

Ez a függvény végzi a megjelenítést egy külön ablakban. A feldolgozás is itt történik a YOLO modell használatával.

```python
def display_image(self):
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection", 800, 600)

    while rclpy.ok() and self.running:
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                self.latest_frame = None

        if frame is not None:
            # Objektumdetektálás
            results = self.model.predict(frame, imgsz=640, conf=0.5)

            # Eredmények megjelenítése
            annotated_frame = results[0].plot()
            cv2.imshow("Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False
            break

    cv2.destroyAllWindows()
```

### Egyéb függvények

Az osztály tartalmaz még egy `stop(self)` függvényt is a leállításhoz, valamint a `main()` részben inicializáljuk a node-ot.

---

## Turtlebot3 beállítás

A rendszer működéséhez szükség van a `turtlebot3_simulations` csomagra, amely a robot modellt tartalmazza. A használt típus: **burger**.

```bash
export TURTLEBOT3_MODEL=burger
```

### Kamera nézet beállítása

A kamera alapból lefelé néz. Ezt módosítani kell:

**Fájl:**  
`turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf`

**Eredeti sor (378):**

```xml
<pose>0.03 0 0.11 0 0.524 0</pose>
```

**Módosítva:**

```xml
<pose>0.03 0 0.11 0 5.93 0</pose>
```

Ez nagyjából 45°-os szöget ad felfelé.

---

## Futtatás

Ne felejtsd el build-elni és source-olni a workspace-t:

```bash
colcon build
source install/setup.bash
```


### Szimuláció indítása

```bash
ros2 launch object_detection_project turtlebot3_world.launch.py
```

Ez elindítja a világot a robottal.
![alt text][image1]

### Robot mozgatása

Egy másik terminálban:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Objektum detektálás indítása

Új terminálban:

```bash
ros2 run object_detection_py object_detection_node
```

---

## Összefoglalás

Ez a projekt egy Gazebo világban futó Turtlebot3 segítségével végzi objektumok felismerését valós idejű YOLOv8-alapú képfeldolgozással. A fenti lépések végrehajtásával teljes körű szimulációt kapsz, mozgó robottal és élő objektumdetektálással.
