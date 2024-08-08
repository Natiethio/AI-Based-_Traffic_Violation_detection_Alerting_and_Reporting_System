import json
import tkinter as tk
from tkinter import filedialog

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

class TrafficConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Configuration")
        self.data = None

        self.scrollable_frame = ScrollableFrame(self.root)
        self.scrollable_frame.pack(fill="both", expand=True)

        self.create_widgets()

    def create_widgets(self):
        self.red_light_frame = self.create_section_frame("Red Light Line", 0)
        self.create_red_light_entries(self.red_light_frame)

        self.speed_test_frame = self.create_section_frame("Speed Test Line", 1)
        self.create_speed_test_entries(self.speed_test_frame)

        self.license_test_frame = self.create_section_frame("License Test Line", 2)
        self.create_licenseplate_line_entries(self.license_test_frame)
        
        self.lane_frame = self.create_section_frame("Lanes", 3)
        self.create_lane_entries(self.lane_frame)

        self.create_buttons()

        self.status = tk.Label(self.scrollable_frame.scrollable_frame, text="")
        self.status.grid(row=5, column=0, columnspan=2, pady=10)

    def create_section_frame(self, title, row):
        frame = tk.LabelFrame(self.scrollable_frame.scrollable_frame, text=title, padx=10, pady=10)
        frame.grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        return frame
    
    def create_red_light_entries(self, frame):
        self.red_line_start_x = self.create_entry(frame, "Red Line Start X:", 0, 0)
        self.red_line_start_y = self.create_entry(frame, "Red Line Start Y:", 0, 1)
        self.red_line_end_x = self.create_entry(frame, "Red Line End X:", 1, 0)
        self.red_line_end_y = self.create_entry(frame, "Red Line End Y:", 1, 1)

    def create_licenseplate_line_entries(self, frame):
        self.licenseplate_line_start_x = self.create_entry(frame, "License Line Start X:", 0, 0)
        self.licenseplate_line_start_y = self.create_entry(frame, "License Line Start Y:", 0, 1)
        self.licenseplate_line_end_x = self.create_entry(frame, "License Line End X:", 1, 0)
        self.licenseplate_line_end_y = self.create_entry(frame, "License Line End Y:", 1, 1)

    def create_speed_test_entries(self, frame):
        self.green_line_start_x = self.create_entry(frame, "Green Line Start X:", 0, 0)
        self.green_line_start_y = self.create_entry(frame, "Green Line Start Y:", 0, 1)
        self.green_line_end_x = self.create_entry(frame, "Green Line End X:", 1, 0)
        self.green_line_end_y = self.create_entry(frame, "Green Line End Y:", 1, 1)
        self.blue_line_start_x = self.create_entry(frame, "Blue Line Start X:", 2, 0)
        self.blue_line_start_y = self.create_entry(frame, "Blue Line Start Y:", 2, 1)
        self.blue_line_end_x = self.create_entry(frame, "Blue Line End X:", 3, 0)
        self.blue_line_end_y = self.create_entry(frame, "Blue Line End Y:", 3, 1)

    def create_lane_entries(self, frame):
        self.lane_count_value = self.create_entry(frame, "Number of Lanes:", 0, 0, columnspan=2)
        self.lanes = []
        self.add_lane_button = tk.Button(frame, text="Add Lane", command=self.add_lane)
        self.add_lane_button.grid(row=1, column=0, pady=5)
        self.remove_lane_button = tk.Button(frame, text="Remove Lane", command=self.remove_lane)
        self.remove_lane_button.grid(row=1, column=1, pady=5)

    def create_entry(self, frame, label_text, row, column, columnspan=1):
        label = tk.Label(frame, text=label_text)
        label.grid(row=row, column=column, padx=5, pady=5, sticky="e")
        entry = tk.Entry(frame)
        entry.grid(row=row, column=column + 1, padx=5, pady=5, columnspan=columnspan, sticky="w")
        return entry

    def create_buttons(self):
        button_frame = tk.Frame(self.scrollable_frame.scrollable_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        self.load_button = tk.Button(button_frame, text="Load JSON", command=self.load_json)
        self.load_button.grid(row=0, column=0, padx=10)
        self.save_button = tk.Button(button_frame, text="Save JSON", command=self.save_json)
        self.save_button.grid(row=0, column=1, padx=10)
    
    def load_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as file:
                self.data = json.load(file)
            self.populate_entries()

    def populate_entries(self):
        if self.data:
            self.clear_entries()
            self.red_line_start_x.insert(0, self.data["red_light_line"]["red_line_start"]["x"])
            self.red_line_start_y.insert(0, self.data["red_light_line"]["red_line_start"]["y"])
            self.red_line_end_x.insert(0, self.data["red_light_line"]["red_line_end"]["x"])
            self.red_line_end_y.insert(0, self.data["red_light_line"]["red_line_end"]["y"])

            self.licenseplate_line_start_x.insert(0, self.data["licenseplate_line"]["licenseplate_line_start"]["x"])
            self.licenseplate_line_start_y.insert(0, self.data["licenseplate_line"]["licenseplate_line_start"]["y"])
            self.licenseplate_line_end_x.insert(0, self.data["licenseplate_line"]["licenseplate_line_end"]["x"])
            self.licenseplate_line_end_y.insert(0, self.data["licenseplate_line"]["licenseplate_line_end"]["y"])

            self.green_line_start_x.insert(0, self.data["speed_test_line"]["green_line_start"]["x"])
            self.green_line_start_y.insert(0, self.data["speed_test_line"]["green_line_start"]["y"])
            self.green_line_end_x.insert(0, self.data["speed_test_line"]["green_line_end"]["x"])
            self.green_line_end_y.insert(0, self.data["speed_test_line"]["green_line_end"]["y"])
            self.blue_line_start_x.insert(0, self.data["speed_test_line"]["blue_line_start"]["x"])
            self.blue_line_start_y.insert(0, self.data["speed_test_line"]["blue_line_start"]["y"])
            self.blue_line_end_x.insert(0, self.data["speed_test_line"]["blue_line_end"]["x"])
            self.blue_line_end_y.insert(0, self.data["speed_test_line"]["blue_line_end"]["y"])

            self.lane_count_value.insert(0, self.data["lane"]["number_of_lanes"])
            self.populate_lanes(self.data["lane"]["lanes"])

    def clear_entries(self):
        self.red_line_start_x.delete(0, tk.END)
        self.red_line_start_y.delete(0, tk.END)
        self.red_line_end_x.delete(0, tk.END)
        self.red_line_end_y.delete(0, tk.END)

        self.licenseplate_line_start_x.delete(0, tk.END)
        self.licenseplate_line_start_y.delete(0, tk.END)
        self.licenseplate_line_end_x.delete(0, tk.END)
        self.licenseplate_line_end_y.delete(0, tk.END)

        self.green_line_start_x.delete(0, tk.END)
        self.green_line_start_y.delete(0, tk.END)
        self.green_line_end_x.delete(0, tk.END)
        self.green_line_end_y.delete(0, tk.END)

        self.blue_line_start_x.delete(0, tk.END)
        self.blue_line_start_y.delete(0, tk.END)
        self.blue_line_end_x.delete(0, tk.END)
        self.blue_line_end_y.delete(0, tk.END)

        self.lane_count_value.delete(0, tk.END)

        for lane in self.lanes:
            lane[0].destroy()
            lane[1].destroy()
        self.lanes.clear()

    def populate_lanes(self, lanes):
        for lane in lanes:
            self.add_lane(lane)

    def add_lane(self, lane=None):
        row = len(self.lanes) + 2  # Adjust row number for lanes
        lane_start_x = self.create_entry(self.lane_frame, "Lane Start X:", row, 0)
        lane_start_y = self.create_entry(self.lane_frame, "Lane Start Y:", row, 1)
        lane_end_x = self.create_entry(self.lane_frame, "Lane End X:", row, 2)
        lane_end_y = self.create_entry(self.lane_frame, "Lane End Y:", row, 3)

        if lane:
            lane_start_x.insert(0, lane["lane_start"]["x"])
            lane_start_y.insert(0, lane["lane_start"]["y"])
            lane_end_x.insert(0, lane["lane_end"]["x"])
            lane_end_y.insert(0, lane["lane_end"]["y"])

        self.lanes.append((lane_start_x, lane_start_y, lane_end_x, lane_end_y))

    def remove_lane(self):
        if self.lanes:
            lane = self.lanes.pop()
            lane[0].destroy()
            lane[1].destroy()
            lane[2].destroy()
            lane[3].destroy()

    def save_json(self):
        data = {
            "red_light_line": {
                "red_line_start": {"x": self.red_line_start_x.get(), "y": self.red_line_start_y.get()},
                "red_line_end": {"x": self.red_line_end_x.get(), "y": self.red_line_end_y.get()}
            },
            "licenseplate_line": {
                "licenseplate_line_start": {"x": self.licenseplate_line_start_x.get(), "y": self.licenseplate_line_start_y.get()},
                "licenseplate_line_end": {"x": self.licenseplate_line_end_x.get(), "y": self.licenseplate_line_end_y.get()}
            },
            "speed_test_line": {
                "green_line_start": {"x": self.green_line_start_x.get(), "y": self.green_line_start_y.get()},
                "green_line_end": {"x": self.green_line_end_x.get(), "y": self.green_line_end_y.get()},
                "blue_line_start": {"x": self.blue_line_start_x.get(), "y": self.blue_line_start_y.get()},
                "blue_line_end": {"x": self.blue_line_end_x.get(), "y": self.blue_line_end_y.get()}
            },
            "lane": {
                "number_of_lanes": self.lane_count_value.get(),
                "lanes": []
            }
        }

        for lane in self.lanes:
            lane_data = {
                "lane_start": {"x": lane[0].get(), "y": lane[1].get()},
                "lane_end": {"x": lane[2].get(), "y": lane[3].get()}
            }
            data["lane"]["lanes"].append(lane_data)

        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            self.status.config(text="Data saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficConfigGUI(root)
    root.mainloop()
