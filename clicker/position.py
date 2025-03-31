from pynput import mouse, keyboard

# Flag to stop the program
running = True

def on_click(x, y, button, pressed):
    if pressed:
        print(f"Mouse clicked at ({x}, {y})")

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        running = False
        return False  # Stop the keyboard listener

# Start the mouse listener
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

# Start the keyboard listener
with keyboard.Listener(on_press=on_press) as keyboard_listener:
    keyboard_listener.join()

# Stop the mouse listener when the program is aborted
mouse_listener.stop()