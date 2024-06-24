from enum import StrEnum
import time
from pathlib import Path
import tkinter as tk
from tkinter.messagebox import showerror
from tkinter.filedialog import askopenfilename
from typing import Optional


from PIL import ImageTk, Image
from PIL.Image import Resampling


from tile_generation import TileGenerationConfig, TileGenerator, generate_tiles


def run_tile_generation(config: TileGenerationConfig, input_image: Path):
    root = tk.Tk()
    app = TileSimulator(root, config, input_image)
    root.mainloop()


class TileErrors(StrEnum):
    MISSING_IMAGE_PATH = "Missing input image path"
    INVALID_IMAGE_PATH = "Invalid input image path"
    CANT_OPEN_IMAGE = "Unhandled exception when opening image"


class TileSimulator:

    CANVAS_SIZE = 400

    def __init__(
        self, root: tk.Tk, config: TileGenerationConfig, input_image: Optional[Path]
    ):
        self.master = root
        self.config = config
        self.input_image = input_image
        self.is_paused = True
        self.tile_generator: Optional[TileGenerator] = None

        controls_frame = tk.Frame(root)

        self.path_label = tk.Label(controls_frame, text=f"Input: {self.input_image}")
        self.path_label.pack()

        button_frame = tk.Frame(controls_frame)

        self._pause_button = tk.Button(
            button_frame, text="play", command=self.toggle_pause
        )
        self._pause_button.pack(side=tk.LEFT)

        choose_button = tk.Button(
            button_frame, text="Choose Image", command=self.prompt_file
        )
        choose_button.pack(side=tk.LEFT)

        reset = tk.Button(button_frame, text="Reset", command=self.prompt_file)
        reset.pack(side=tk.LEFT)

        button_frame.pack()
        controls_frame.pack()

        # Canvas doesn't work for unknown reasons
        # self.tag: Optional[int] = None
        # self.canvas = tk.Canvas(width=self.CANVAS_SIZE, height=self.CANVAS_SIZE)
        # self.canvas.pack()

        self.img_container = tk.Label(root)
        self.img_container.pack()

    def run(self, new_image: Optional[Path] = None):
        if new_image is not None:
            self.input_image = new_image

        self._setup_tiler()
        self._step_tiler()

    def reset(self):
        self.is_paused = True

        if self.tile_generator:
            self.tile_generator.reset()

    def redraw(self):
        # Pause button
        text = "play" if self.is_paused else "pause"
        self._pause_button.config(text=text)

        # Path
        self.path_label.config(text=f"Input: {self.input_image}")

        # Tiles
        if self.tile_generator is not None:
            self.raw_image = self._fit_image(self.tile_generator.build_image())
            self._draw_image(self.raw_image)

    def prompt_file(self):
        self.is_paused = True

        file_name = askopenfilename(title="Input Image File")
        path = Path(file_name)
        self.input_image = path
        self.reset()
        self.redraw()

    def toggle_pause(self):
        self.is_paused = not self.is_paused

        if not self.is_paused:
            self.run()

        self.redraw()

    def _setup_tiler(self):
        if self.input_image is None:
            showerror(title="Error", message=TileErrors.MISSING_IMAGE_PATH)
            return

        if not self.input_image.exists() or not self.input_image.is_file():
            showerror(message=TileErrors.INVALID_IMAGE_PATH)
            return

        try:
            image = Image.open(self.input_image)
        except:
            showerror(message=TileErrors.CANT_OPEN_IMAGE)
            return

        self.tile_generator = TileGenerator(self.config, image)

    def _step_tiler(self):
        if not self.tile_generator:
            return

        if self.tile_generator.status != "Pending":
            self.redraw()
            return

        print("STEPPING")
        self.tile_generator.step()
        self.redraw()

        self.master.after(20, self._step_tiler)

    def _save_image(self, image: Image.Image):
        with open("./results/test.jpg", "w") as fp:
            image.save(fp)

    def _fit_image(self, image: Image.Image):
        return image.resize(
            (self.CANVAS_SIZE * 2, self.CANVAS_SIZE * 2), Resampling.NEAREST
        )

    def _draw_image(self, image: Image.Image):
        self._image_tk = ImageTk.PhotoImage(image, (20, 20))
        self.img_container.config(image=self._image_tk)
