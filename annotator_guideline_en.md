# 3D Annotation Tool — Annotator Guideline

## Getting started

1. Choose your **Dataset**, **Sequence**, and **Camera Channel** from the dropdowns at the top right.
2. The main window shows the 3D point cloud. Camera images for the selected channel and a Bird's Eye View (BEV) tile appear along the top.
3. Navigate frames with the **◀ / ▶** buttons or `P` / `N`. The frame counter (e.g. `4/150`) is at the bottom.
4. Press `F` any time to open the frame info panel — useful for confirming which image/annotation/lidar file you're on.
![Overview](assets/guideline/1.png)

## Creating a box

1. Pick the object class from the class picker on the left, or press `0`–`9`. 

![Class picker](assets/guideline/2.png)

2. In the **Bird's Eye View** or 3D window, hold `Ctrl` and drag across the object on the ground to draw a box. 
3. A new box appears with default size for that class — adjust it to fit the object accurately.


## Selecting and adjusting a box

**Select:** left-click the box in the 3D view, or use `Tab` / `Shift+Tab` to cycle through all boxes in the frame.
- Once selected, side/front/BEV close-up views appear to help with fine adjustment.
**Switch adjustment mode:**
  - `Alt` + `T` → Translate (move)
  - `Alt` + `R` → Rotate
  - `Alt` + `S` → Scale (resize)
**Nudge with keyboard** (translate mode): `W`/`A`/`S`/`D` to move, `Q`/`E` for the third axis.
- Hold `Ctrl` while dragging a gizmo handle to snap to 0.5 m position / 15° rotation increments.

## Panel
Right-hand panel include frame's panel and box panels.

| Frame's panel | Box's panel |
|---|---|
|![Frame attribute](assets/guideline/5.png) | ![Box attribute](assets/guideline/4.png)|


## Attributes

- Set the frame's **Weather Type** from the dropdown in the frame's panel — this applies to the whole frame, not a single box.
- Open the box's folder in the box's panel to set **Visibility** and other class-specific attributes.

## Copying labels across frames

- Check **"Copy label to next frame"** on a box's panel before moving to the next frame to carry it forward automatically.
- Use **"Select all / Unselect all copy label to next frame"** at the frame's panel to apply this to every box at once.

## Deleting

- `Delete` / `Backspace`, or `Ctrl` + right-click a box: delete it in the current frame only.
- "Delete in all frames" (in the box's panel): removes that tracked object everywhere.

## Undo & saving

- `Ctrl` + `Z`: undo the last action (works for delete, move, rotate, scale, track ID changes, frame changes).
- The tool **autosaves the current frame every 5 seconds** and also on frame/sequence/dataset switches — you don't need to save manually, but don't close the tab mid-edit on a frame you just changed.

## Quick reference

| Key | Action |
|---|---|
| `N` / `P` | Next / previous frame |
| `Tab` / `Shift+Tab` | Next / previous object |
| `0`–`9` | Select class |
| `Alt+T` / `Alt+R` / `Alt+S` | Translate / Rotate / Scale mode |
| `Space` | Play/pause sequence (nothing selected) |
| `Ctrl`+drag (BEV) | Draw new box |
| `Delete` | Delete selected box |
| `Ctrl+Z` | Undo |
| `C` | Switch orthographic/perspective view |
| `F` | Toggle frame info panel |
| `Esc` | Deselect |

**When in doubt:** select the object, check the side/front/BEV close-ups line up with the point cloud, and confirm the class and attributes are correct before moving to the next frame.
![](assets/guideline/3.png)
