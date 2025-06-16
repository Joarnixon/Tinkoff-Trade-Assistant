import tkinter as tk
import sys
import asyncio
import websockets
import json
from datetime import datetime, timezone
import requests
from collections import deque
import pandas as pd
from threading import Thread
import sys
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import uuid
import mplfinance as mpf

class ChartDataQueue:
    def __init__(self, maxlen=119):
        self.maxlen = maxlen
        self.queue = deque(maxlen=maxlen)
        self.df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df.index = pd.to_datetime(self.df.index)
        self.df.index.name = 'Date'

    def append(self, data):
        # Validation: Check if the new data point is different from the last one
        if self.queue and data['Date'] == self.queue[-1]['Date']:
            return 

        self.queue.append(data)
        self.update_dataframe()

    def update_dataframe(self):
        self.df = pd.DataFrame(list(self.queue), columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df.index = pd.to_datetime([item['Date'] for item in self.queue])
        self.df.index.name = 'Date'

    def clear(self):
        self.queue.clear()
        self.update_dataframe()

    @property
    def empty(self):
        return len(self.queue) == 0

class WebSocketApp:
    def __init__(self, master):
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.websocket = None
        master.title("Stock Viewer")

        self.is_running = True
        self.websocket = None
        self.loop = None
        self.websocket_thread = None

        self.master.geometry("1200x600")

        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame for chart
        left_frame = tk.Frame(main_frame, width=800, height=600)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right frame for share list
        right_frame = tk.Frame(main_frame, width=400, height=600)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Status label
        self.status_label = tk.Label(left_frame, text="Connecting...")
        self.status_label.pack()

        # Canvas for stock chart
        self.chart_canvas = tk.Canvas(left_frame, bg="white")
        self.chart_canvas.pack(fill=tk.BOTH, expand=True)

        # Search entry
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.schedule_update_shares_list)
        search_entry = tk.Entry(right_frame, textvariable=self.search_var)
        search_entry.pack(fill=tk.X, padx=5, pady=5)
        self.update_timer = None
        self.search_entry = search_entry

        # Scrollable frame for share list
        self.shares_canvas = tk.Canvas(right_frame)
        self.shares_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.shares_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.shares_canvas.configure(yscrollcommand=scrollbar.set)
        self.shares_canvas.bind('<Configure>', lambda e: self.shares_canvas.configure(scrollregion=self.shares_canvas.bbox("all")))

        self.shares_frame = tk.Frame(self.shares_canvas)
        self.shares_canvas.create_window((0, 0), window=self.shares_frame, anchor="nw")

        self.shares = self.fetch_shares()
        self.share_widgets = []
        self.selected_shares = {}
        self.update_shares_list()

        # Generate a safe user ID
        self.user_id = str(uuid.uuid4())
        # Chart data
        self.chart_data = ChartDataQueue(maxlen=119)
        self.update_id = None
        self.clicked_figi = None
        self.status_timer = None

        # To store the recieved predictions
        self.predictions = {}
        self.max_subscriptions = 3  # Maximum allowed subscriptions, also limited on the server side
        
        # Create the initial chart
        self.create_chart()

        # Start the WebSocket connection
        self.start_websocket()

    def show_temporary_status(self, message, duration=5000):
        """Show a temporary status message."""
        # Cancel any existing timer
        if self.status_timer:
            self.master.after_cancel(self.status_timer)
        
        # Save the current status
        current_status = self.status_label.cget("text")
        
        # Update status with the new message
        self.master.after(0, self.safe_update, self.status_label, "text", message)

        # Set a timer to revert the status
        self.status_timer = self.master.after(0, self.safe_update, self.status_label, "text", current_status)

    def fetch_shares(self):
        try:
            response = requests.get('http://127.0.0.1:8000/shares/')
            if response.status_code == 200:
                shares_dict = response.json()
                return [{"name": name, "info": "", "figi": figi} for figi, name in shares_dict.items()]
            else:
                print(f"Failed to fetch shares. Status code: {response.status_code}")
                return []
        except Exception as e:
            print
            
    def create_chart(self):
        focused_widget = self.master.focus_get()

        # Clear any existing chart
        for widget in self.chart_canvas.winfo_children():
            widget.destroy()

        # Always create a Figure and FigureCanvasTkAgg
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
        self.chart_widget = FigureCanvasTkAgg(fig, master=self.chart_canvas)
        self.chart_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if not self.chart_data.empty:
            additional_plots = self.get_plots(ax)
            vlines = self.get_prediction_vlines()
            if vlines:
                mpf.plot(self.chart_data.df, type='candle', style='charles', ax=ax, 
                        addplot=additional_plots, vlines=vlines)
            else:
                mpf.plot(self.chart_data.df, type='candle', style='charles', ax=ax, addplot=additional_plots)
        else:
            ax.text(0.5, 0.5, "No chart data available.", 
                    horizontalalignment='center', 
                    verticalalignment='center', 
                    transform=ax.transAxes)        
        fig.tight_layout(rect=(0, 0.05, 0.93, 1), pad=0)

        self.chart_widget.draw()

        self.chart_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Restore focus to the previously focused widget
        if focused_widget and focused_widget.winfo_exists(): 
            self.master.after(10, lambda: focused_widget.focus_set())
        
    def update_chart(self):
        if self.update_id:
            self.master.after_cancel(self.update_id)

        figi = self.clicked_figi

        try:
            response = requests.get(f"http://127.0.0.1:8000/chart/{figi}/{len(self.chart_data.queue)}")
            if response.status_code == 200:
                new_data = response.json()
                self.add_candle_to_chart(new_data)

                # Force redraw after adding data:
                self.redraw_chart() 

        except requests.RequestException as e:
            print(f"Error fetching chart data: {e}")

        self.update_id = self.master.after(10000, self.update_chart)

    def add_candle_to_chart(self, candle_data):
        if not candle_data:  # Early exit if candle_data is empty
            return

        for candle in candle_data:
            new_data = {
                'Date': pd.to_datetime(candle['time'], format='%Y-%m-%dT%H:%M:%S%z'),
                'Open': candle['open']['units'] + candle['open']['nano'] / 1e9,
                'High': candle['high']['units'] + candle['high']['nano'] / 1e9,
                'Low': candle['low']['units'] + candle['low']['nano'] / 1e9,
                'Close': candle['close']['units'] + candle['close']['nano'] / 1e9,
                'Volume': candle['volume']
            }
            self.chart_data.append(new_data)

    def redraw_chart(self):
        if hasattr(self, 'chart_widget'):
            self.chart_widget.figure.clf()
            ax = self.chart_widget.figure.add_subplot(111)

            # Check if chart_data is empty BEFORE plotting:
            if self.chart_data.empty:
                # Display "No data" message on the axes
                ax.text(0.5, 0.5, "No chart data available.", 
                        horizontalalignment='center', 
                        verticalalignment='center', 
                        transform=ax.transAxes)
                ax.axis('off')
            else:
                vlines = self.get_prediction_vlines()
                if vlines:
                    mpf.plot(self.chart_data.df, type='candle', style='charles', ax=ax, 
                            addplot=self.get_plots(ax), vlines=vlines)
                else:
                    mpf.plot(self.chart_data.df, type='candle', style='charles', ax=ax, addplot=self.get_plots(ax))

            self.chart_widget.draw()
        else:
            self.create_chart()

    def show_chart(self, event, figi):
        if self.selected_shares.get(figi, False) or sum(self.selected_shares.values()) != self.max_subscriptions:
            self.chart_data.clear()
            if self.update_id:
                self.master.after_cancel(self.update_id)
            if self.clicked_figi == figi:
                self.create_chart()
                self.clicked_figi = None
            else:
                self.clicked_figi = figi
                self.update_chart()
                self.redraw_chart()
        else:
            self.show_temporary_status(f"You can only subscribe to a maximum of {self.max_subscriptions} shares.")
            return
    
    def get_plots(self, ax):
        plots = []
        if len(self.chart_data.queue) >= 5:  # Make sure we have enough data for the MA
            ma_plot = mpf.make_addplot(self.chart_data.df['Close'].rolling(window=5).mean(), ax=ax, type='line', color='black')
            plots.append(ma_plot)
        return plots
    
    def add_share(self, name, info, figi, selected=False):
        frame = tk.Frame(self.shares_frame, bg='SystemButtonFace', width=380, height=50)  # Set a fixed width
        frame.pack(fill=tk.X, padx=5, pady=5)
        frame.pack_propagate(False)  # Prevent the frame from resizing based on its contents

        text_frame = tk.Frame(frame, bg='SystemButtonFace')
        text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Truncate the name if it's too long
        max_name_length = 15
        truncated_name = name[:max_name_length] + '...' if len(name) > max_name_length else name
        
        name_label = tk.Label(text_frame, text=truncated_name, font=("Arial", 12, "bold"), bg='SystemButtonFace', anchor="w")
        name_label.pack(fill=tk.X)
        
        # Info label is now empty but kept for consistency
        info_label = tk.Label(text_frame, text="", wraplength=200, bg='SystemButtonFace')
        info_label.pack(anchor="w")

        subscribe_label = tk.Label(frame, text="Subscribe" if not selected else "Unsubscribe", width=10, fg='#666666', font=("Arial", 10), bg='SystemButtonFace', cursor='hand2')
        subscribe_label.pack(side=tk.RIGHT, padx=5)
        subscribe_label.bind("<Button-1>", lambda event, f=figi: self.toggle_subscription(f))

        color = 'lightgray' if selected else 'SystemButtonFace'

        for widget in (frame, text_frame, name_label, info_label, subscribe_label):
            widget.config(bg=color)

        for widget in (frame, text_frame, name_label, info_label):
            widget.bind("<Button-1>", lambda event, f=figi: self.show_chart(event, f))

        self.share_widgets.append((frame, text_frame, name_label, info_label, figi, selected, subscribe_label))
        self.selected_shares[figi] = selected

    def toggle_subscription(self, figi):
        current_subscriptions = sum(self.selected_shares.values())
        
        for i, (frame, text_frame, name_label, info_label, f, selected, subscribe_button) in enumerate(self.share_widgets):
            if f == figi:
                new_state = not selected
                
                # the limit of subscriptions, also limited on the server side
                if new_state and current_subscriptions >= self.max_subscriptions:
                    self.show_temporary_status(f"You can only subscribe to a maximum of {self.max_subscriptions} shares.")
                    return

                self.selected_shares[figi] = new_state
                self.clicked_figi = figi
                self.chart_data.clear()

                if new_state:
                    color = 'lightgray'
                    subscribe_button.config(text="Unsubscribe")
                    self.update_chart()
                else:
                    color = 'SystemButtonFace'
                    subscribe_button.config(text="Subscribe")
                    
                for widget in (frame, text_frame, name_label, info_label, subscribe_button):
                    widget.config(bg=color)

                self.share_widgets[i] = (frame, text_frame, name_label, info_label, f, new_state, subscribe_button)
                
                if self.websocket:
                    asyncio.run_coroutine_threadsafe(self.send_figi(figi, new_state), self.loop)
                
                # Update only the affected widget
                frame.update_idletasks()
                break
        
    def schedule_update_shares_list(self, *args):
        if self.update_timer is not None:
            self.master.after_cancel(self.update_timer)
        self.update_timer = self.master.after(300, self.update_shares_list)  # 300 ms delay

    def update_shares_list(self, *args):
        for widget, _, _, _, _, _, _ in self.share_widgets:
            widget.pack_forget()

        self.share_widgets = []

        search_term = self.search_var.get().lower()

        for share in self.shares:
            if search_term in share['name'].lower():
                figi = share['figi']
                selected = self.selected_shares.get(figi, False) 
                self.add_share(share['name'], share['info'], figi, selected)

        self.shares_frame.update_idletasks()
        self.shares_canvas.configure(scrollregion=self.shares_canvas.bbox("all"))

    
    async def send_figi(self, figi, state):
        if self.websocket:
            await self.websocket.send(json.dumps({'FIGI': figi, 'STATE': state}))
    
    def get_prediction_vlines(self):
        if self.clicked_figi not in self.predictions:
            return None
        if not self.predictions[self.clicked_figi]:
            return None
        
        vlines = []
        colors = []
        linewidths = []
        alphas = []

        chart_start = self.chart_data.df.index[0]
        chart_end = self.chart_data.df.index[-1]
        
         # Calculate the width of a single candle
        chart_width = self.chart_canvas.winfo_width()
        num_candles = len(self.chart_data.df)
        candle_width = chart_width / num_candles
        
        # Adjust linewidth based on candle width, with a minimum and maximum
        linewidth = max(1, candle_width * 0.8)

        for i, (pred_time, prediction, proba) in enumerate(self.predictions[self.clicked_figi]):
            if prediction == 0:
                continue
            
            if pred_time <= chart_end:
                vlines.append(pred_time)
            elif pred_time > chart_end:
                vlines.append(chart_end)
            elif pred_time < chart_start:
                self.predictions[self.clicked_figi].pop(i)
                continue
            color = 'green' if prediction == 1 else 'red'
            colors.append(color)
            linewidths.append(linewidth)
            alphas.append(proba * 0.5)
        
        if not vlines:  # If no valid predictions within the chart range
            return None
        
        return dict(
            vlines=vlines,
            colors=colors,
            linewidths=linewidths,
            alpha=alphas
        )

    async def handle_prediction(self, figi, prediction, proba):
        if figi in self.selected_shares and self.selected_shares[figi]:
            current_time = datetime.now(timezone.utc)
            if figi not in self.predictions:
                self.predictions[figi] = []
            self.predictions[figi].append((current_time, prediction, proba))
            if figi == self.clicked_figi:
                self.master.after(0, self.redraw_chart)  # Schedule redraw on the main thread

    def toggle_connection(self):
        if self.websocket_thread is None or not self.websocket_thread.is_alive():
            self.start_websocket()
        else:
            self.stop_websocket()

    def start_websocket(self):
        self.is_running = True
        self.websocket_thread = Thread(target=self.run_websocket_loop, args=(self.user_id,))
        self.websocket_thread.start()

    def stop_websocket(self):
        if self.loop and self.is_running:
            self.is_running = False
            asyncio.run_coroutine_threadsafe(self.close_websocket(), self.loop)
        self.connect_button.config(text="Connect")

    def run_websocket_loop(self, user_id):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.websocket_client(user_id))
        self.loop.close()

    def run_websocket(self, user_id):
        asyncio.run(self.websocket_client(user_id))

    async def websocket_client(self, user_id):
        uri = f"ws://127.0.0.1:8000/ws/{user_id}"
        try:
            async with websockets.connect(uri) as websocket:
                self.websocket = websocket
                self.master.after(0, self.safe_update, self.status_label, "text", "Connected")
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.5)

                        message = json.loads(json.loads(message))
                        figi = next(iter(message))
                        prediction, proba = message[figi].values()
                        prediction = int(prediction[0])
                        proba = proba[0][prediction]
                        await self.handle_prediction(figi, prediction, proba)
                    except asyncio.TimeoutError:
                        continue 
        except websockets.exceptions.ConnectionClosed:
            self.master.after(0, self.safe_update, self.status_label, "text", "Connection closed")
        except Exception as e:
            self.master.after(0, self.safe_update, self.status_label, "text", f"Error: {str(e)}")
        finally:
            self.websocket = None

    async def close_websocket(self):
        if self.websocket:
            await self.websocket.close()
        self.websocket = None

    def safe_update(self, widget, attribute, value):
        if self.is_running and widget.winfo_exists():
            widget.config(**{attribute: value})

    def on_closing(self):
        self.is_running = False
        if self.loop:
            tasks = []
            if self.websocket:
                tasks.append(self.close_websocket())
            tasks.append(self.loop.shutdown_asyncgens())
            tasks.append(self.loop.shutdown_default_executor())
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop.call_soon_threadsafe(self.loop.create_task, asyncio.gather(*tasks, return_exceptions=True))
        if self.websocket_thread and self.websocket_thread.is_alive():
            self.websocket_thread.join(timeout=1)
            
        self.master.destroy()
        root.quit()
        sys.exit(0)

root = tk.Tk()
app = WebSocketApp(root)
root.mainloop()