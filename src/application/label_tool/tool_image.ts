import * as THREE from "three";
import {Utils} from "../util/utils";
import * as $ from "jquery";
import {MathUtils} from "../util/math_utils";
import {LabelTool} from "./tool_main";
import {AnnotationClass} from "../annotation/annotation_class";
import {AnnotationObject, AnnotationObjectParams} from "../annotation/annotation_object";
import {RaphaelElement, RaphaelPaper, RaphaelPath} from "raphael";
import {Vector2, Vector3} from "three";

class LabelToolImage{
    labelTool: LabelTool;

    annotationClasses: AnnotationClass;
    annotationObjects: AnnotationObject;

    canvasArray: HTMLCanvasElement[] = [];
    canvasParamsArray: any[] = [];

    paperArray: RaphaelPaper[] = [];
    paperArrayAll: RaphaelPaper[][] = [];

    imageArray: RaphaelElement<"SVG" | "VML", Element | SVGImageElement>[] = [];
    imageArrayAll: RaphaelElement<"SVG" | "VML", Element | SVGImageElement>[][] = [];
    fontSize: number = 20;
    action: string = "add";
    mouseX: number = 0;
    mouseY: number = 0;
    imageWidthOriginal: number = -1;
    imageHeightOriginal: number = -1;
    headerHeight: number = 0;
    circleArray: RaphaelElement<"SVG" | "VML", Element | SVGCircleElement>[] = [];

    // BEV is a single top-down panel per frame (not one per camera
    // channel), so it gets its own Raphael paper/image arrays indexed
    // only by fileIndex.
    paperArrayAllBEV: RaphaelPaper[] = [];
    imageArrayAllBEV: RaphaelElement<"SVG" | "VML", Element | SVGImageElement>[] = [];
    // The single "image-bev" DOM canvas (set once in tool_main.ts'
    // initCameraWindows(), analogous to canvasArray[channelIdx] for the
    // per-channel panels).
    canvasElemBEV: HTMLCanvasElement | undefined;

    // Pixel size of the images_BEV/*.jpg tiles, and the real-world extent
    // (meters) each tile covers — matches the defaults used to render
    // those images server-side (see build_bev_detection /
    // draw_bev_canvas_world_topview in project_bbox2img.py: w_img=2000,
    // h_img=2250, width_size=20, height_size=30). If the box ends up
    // offset from the vehicle footprint in the rendered tile, this is the
    // first place to check against however images_BEV/ was actually generated.
    readonly bevImageWidth: number = 2000;
    readonly bevImageHeight: number = 2250;
    readonly bevWidthMeters: number = 20;
    readonly bevHeightMeters: number = 30;

    constructor(labelTool: LabelTool, annotationClasses: AnnotationClass, annotationObjects: AnnotationObject) {
        this.labelTool = labelTool;
        this.annotationClasses = annotationClasses;
        this.annotationObjects = annotationObjects;
        this.annotationObjects.setLabelToolImage(this);
        this.imageWidthOriginal = this.labelTool.originalImageSize[0];
        this.imageHeightOriginal = this.labelTool.originalImageSize[1];
    }

    initializeCamChannel(camChannel: string){
        const canvas: HTMLCanvasElement = this.canvasArray[Utils.getChannelIndexByName(this.labelTool.cameraChannels, camChannel)];
        if (canvas !== undefined) {
            let channelIdx = Utils.getChannelIndexByName(this.labelTool.cameraChannels, camChannel);
            this.canvasParamsArray[channelIdx] = {
                x: canvas.offsetLeft,
                y: canvas.offsetTop,
                width: canvas.offsetWidth,
                height: canvas.offsetHeight,
                center: {x: canvas.offsetWidth / 3, y: canvas.offsetHeight / 3}
            };
        }
    }

    loadCameraImages(camChannel: string, fileIndex: number, labelTool){
        let imgPath = "";

        let path_part_1 = "../../input/" + labelTool.currentDataset + "/" + labelTool.currentSequence + "/images/";
        let path_part_2 = camChannel + "/" + labelTool.imageFileNames[camChannel][fileIndex];
        imgPath =  path_part_1 + path_part_2;

        let channelIdx = Utils.getChannelIndexByName(labelTool.cameraChannels, camChannel);

        const paper: RaphaelPaper = this.paperArrayAll[fileIndex][channelIdx];
        this.imageArray[channelIdx] = paper.image(
            imgPath,
            this.labelTool.currentImageArray[channelIdx]['x'],
            this.labelTool.currentImageArray[channelIdx]['y'],
            this.labelTool.currentImageArray[channelIdx]['width'],
            this.labelTool.currentImageArray[channelIdx]['height']);
    }

    /**
     * Loads the frame's top-down BEV image from images_BEV/, which sits
     * next to images/ (not nested inside a per-camera-channel subfolder —
     * BEV is one image per frame, not per channel). Mirrors
     * loadCameraImages' path-building pattern exactly.
     *
     * The paper's coordinate space is the panel's on-screen size
     * (labelTool.canvasSizeBEV), not the tile's native 2000x2250
     * resolution — the image is drawn scaled down to fit, and
     * drawBoundingBoxBEV() scales its box coordinates down the same way
     * so the two line up.
     *
     * Assumes labelTool exposes a per-frame BEV filename list the same
     * way it exposes imageFileNames[camChannel][fileIndex] for camera
     * channels — e.g. labelTool.imageFileNamesBEV[fileIndex]. That list
     * needs to be populated wherever imageFileNames itself gets built
     * (dataset/sequence loading, outside this file) since BEV filenames
     * don't necessarily share a base name with any camera channel's files.
     *
     * Also assumes a #canvasBEV-equivalent Raphael paper already exists
     * at this.paperArrayAllBEV[fileIndex] — create it the same way the
     * per-channel canvases/papers get created (not in this file; see
     * wherever paperArrayAll[fileIndex][channelIdx] is set up) before
     * calling this.
     */
    loadBEVImage(fileIndex: number, labelTool) {
        const bevFileName = labelTool.imageFileNamesBEV?.[fileIndex];
        if (bevFileName === undefined) {
            console.warn("loadBEVImage: labelTool.imageFileNamesBEV[" + fileIndex + "] is not set — populate it wherever imageFileNames is built for camera channels.");
            return;
        }

        const imgPath = "../../input/" + labelTool.currentDataset + "/" + labelTool.currentSequence
            + "/images_BEV/" + bevFileName;

        const paper: RaphaelPaper = this.paperArrayAllBEV[fileIndex];
        if (paper === undefined) {
            console.warn("loadBEVImage: paperArrayAllBEV[" + fileIndex + "] is not set — create the BEV canvas/Raphael paper before calling this.");
            return;
        }

        this.imageArrayAllBEV[fileIndex] = paper.image(
            imgPath,
            0, 0,
            labelTool.canvasSizeBEV[0],
            labelTool.canvasSizeBEV[1],
        );
    }

    cancelDefault(e) {
        e = e || window.event;
        if (e.stopPropagation) e.stopPropagation();
        if (e.preventDefault) e.preventDefault();
        e.cancelBubble = false;
        return false;
    }

    addEvent(element, trigger, action) {
        if (typeof element === "string") {
            element = document.getElementById(element);
        }
        if (element.addEventListener) {
            element.addEventListener(trigger, action, false);
            return true;
        } else if (element.attachEvent) {
            element['e' + trigger + action] = action;
            element[trigger + action] = function () {
                element['e' + trigger + action](window.event);
            };
            let r = element.attachEvent('on' + trigger, element[trigger + action]);
            return r;
        } else {
            element['on' + trigger] = action;
            return true;
        }
    }

    setCursor(cursorType) {
        for (let img in this.imageArray) {
            let imgObj = this.imageArray[img];
            imgObj.attr({cursor: cursorType});
        }
    }



    remove(index: number) {
        this.removeTextBox(index);
    }

    removeTextBox(index) {
        let bbox = this.annotationObjects.contents[this.labelTool.currentFrameIndex][index];
        if (bbox["textBox"] === undefined) {
            return;
        }
        bbox["textBox"]["text"].remove();
        bbox["textBox"]["box"].remove();
        delete bbox["textBox"];
    }


    removeProjectedBoundingBox(channelObj) {
        for (const line of channelObj.lines) {
            if (line !== undefined) {
                line.remove();
            }
        }
    }

    /**
     * Draws a box's footprint (4 bottom-corner edges) on the BEV panel as
     * a simple orthographic top-down projection — unlike the camera
     * channels, BEV has no lens distortion to account for, so this is
     * plain 2D geometry with no server round-trip.
     *
     * Coordinate convention: "depth" (vertical axis in the image) tracks
     * box.x (forward), "left" (horizontal axis) tracks box.y (left), both
     * mirrored around the image center — matches how images_BEV/ is
     * actually rendered (verified visually; boxes appeared flipped
     * top-to-bottom and left-to-right against the tile before the mirror
     * was added). If the box ever renders offset/rotated again after
     * images_BEV/'s generation changes, this mapping is the first place
     * to adjust.
     *
     * Positions are computed in the BEV tile's native 2000x2250 pixel
     * space, then scaled down to the panel's actual on-screen size
     * (labelTool.canvasSizeBEV) — the paper itself is created at that
     * display size (see initCameraWindows in tool_main.ts), matching how
     * loadBEVImage() draws the image scaled down too.
     */
    drawBoundingBoxBEV(box: AnnotationObjectParams, isSelected: boolean, fileIndex: number = this.labelTool.currentFrameIndex) {
        const paper: RaphaelPaper = this.paperArrayAllBEV[fileIndex];
        if (paper === undefined) {
            return;
        }

        const bevBox = (<any>box);
        if (bevBox.bev !== undefined) {
            this.removeProjectedBoundingBox(bevBox.bev);
        }

        const color = isSelected
            ? this.labelTool.colorSelectedObject
            : this.annotationClasses.annotationClasses[box.class].color;

        // 4 bottom corners in box-local space, same consecutive winding
        // order used everywhere else in this codebase (see
        // _corner_points() in project_bbox2img.py) — edges 0-1, 1-2, 2-3,
        // 3-0 form the rectangle.
        const cornersLocal: [number, number][] = [
            [box.length / 2, box.width / 2],
            [box.length / 2, -box.width / 2],
            [-box.length / 2, -box.width / 2],
            [-box.length / 2, box.width / 2],
        ];

        const cosYaw = Math.cos(box.rotationYaw);
        const sinYaw = Math.sin(box.rotationYaw);

        const wRes = this.bevImageWidth / this.bevWidthMeters;
        const hRes = this.bevImageHeight / this.bevHeightMeters;

        // Scale from native tile pixels down to the panel's displayed
        // size. Normally scaleX === scaleY (setCanvasSizeBEV preserves the
        // tile's aspect ratio), kept separate defensively in case that
        // ever changes.
        const scaleX = this.labelTool.canvasSizeBEV[0] / this.bevImageWidth;
        const scaleY = this.labelTool.canvasSizeBEV[1] / this.bevImageHeight;

        const toPixel = (localX: number, localY: number): Vector2 => {
            const worldX = box.x + (localX * cosYaw - localY * sinYaw);
            const worldY = box.y + (localX * sinYaw + localY * cosYaw);
            const depth = worldX;
            const left = worldY;
            // Mirrored on both axes (subtract from center instead of add)
            // relative to the original mapping — matches how images_BEV/
            // is actually rendered; boxes appeared flipped top-to-bottom
            // and left-to-right against the tile before this adjustment.
            return new Vector2(
                (this.bevImageWidth / 2 - left * wRes) * scaleX,
                (this.bevImageHeight / 2 - depth * hRes) * scaleY,
            );
        };

        const pixelCorners = cornersLocal.map(([lx, ly]) => toPixel(lx, ly));

        const lines: RaphaelPath<"SVG" | "VML">[] = [];
        for (let i = 0; i < 4; i++) {
            const p1 = pixelCorners[i];
            const p2 = pixelCorners[(i + 1) % 4];
            let line = paper.path(["M", p1.x, p1.y, "L", p2.x, p2.y] as any);
            line.attr({stroke: color, "stroke-width": 2});
            lines.push(line);
        }

        bevBox.bev = { lines };
    }

    remove2DBoundingBoxes() {
        const frameContents = this.annotationObjects.contents[this.labelTool.currentFrameIndex];
        for (const obj of frameContents) {
            for (const channelObj of obj.channels) {
                for (const line of channelObj.lines) {
                    if (line !== undefined) {
                        line.remove();
                    }
                }
            }

            const bev = (<any>obj).bev;
            if (bev !== undefined) {
                this.removeProjectedBoundingBox(bev);
            }
        }
    }

    // tool_image.ts

    /**
     * Projects one or more 3D boxes into all camera channels in a single request.
     * Returns a map: channelName → array of Vector2[] (one entry per input box).
     */

    async projectBoundingBoxes(
        boxes: Array<{
            x: number;
            y: number;
            z: number;
            length: number;
            width: number;
            height: number;
            yaw: number
        }>
    ): Promise<{ [channel: string]: (Vector2 | undefined)[][] }> {

        const channels = this.labelTool.cameraChannels.map((ch, idx) => ({
            channel: ch.channel,
            zoomFactor: this.labelTool.imageScale[idx],
        }));

        const requestBody = JSON.stringify({
            boxes,
            coordinateSystem: this.labelTool.coordinateSystem,
            channels,
        });

        const response = await fetch('/project_bounding_box', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: requestBody
        });

        if (!response.ok) {
            throw new Error(
                'Projection request failed: ' + response.status
            );
        }

        const result: {
            [channel: string]: (number[] | null)[][]
        } = await response.json();

        const converted: {
            [channel: string]: (Vector2 | undefined)[][]
        } = {};

        for (const channel in result) {

            // Corners outside the camera's calibrated FOV come back as
            // `null` (not a faithful [x,y] projection) instead of being
            // silently folded into a distorted position. Map those to
            // `undefined` so drawLine()'s existing
            // `pointStart !== undefined && isFinite(...)` guard skips any
            // line segment touching them, with no other changes needed.
            converted[channel] =
                result[channel].map(
                    boxPoints =>
                        boxPoints.map(
                            pt => pt === null ? undefined : new THREE.Vector2(
                                pt[0],
                                pt[1]
                            )
                        )
                );
        }

        return converted;
    }


    calculateAndDrawLineSegments(channelObj, className: string, selected: boolean, fileIndex: number = this.labelTool.currentFrameIndex) {
        let channel = channelObj.channel;
        let lineArray: RaphaelPath<"SVG" | "VML">[] = [];
        let channelIdx = Utils.getChannelIndexByName(this.labelTool.cameraChannels, channel);
        let color;
        if (selected === true) {
            color = this.labelTool.colorSelectedObject;
        } else {
            color = this.annotationClasses.annotationClasses[className].color;
        }

        // bottom four lines
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[0], channelObj.projectedPoints[1], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[1], channelObj.projectedPoints[2], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[2], channelObj.projectedPoints[3], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[3], channelObj.projectedPoints[0], color, fileIndex)!);

        // draw line for orientation
        // corners 4-7 may be undefined if that corner fell outside the
        // camera's calibrated FOV (see the null -> undefined mapping in
        // projectBoundingBoxes above) — skip the orientation marker rather
        // than throwing and losing the rest of the box's lines.
        const orientationCorners = [
            channelObj.projectedPoints[6],
            channelObj.projectedPoints[7],
            channelObj.projectedPoints[4],
            channelObj.projectedPoints[5],
        ];
        if (orientationCorners.every(p => p !== undefined)) {
            const pointZero = orientationCorners[0].clone();
            const pointOne = orientationCorners[1].clone();
            const pointTwo = orientationCorners[2].clone();
            const pointThree = orientationCorners[3].clone();

            let startPoint = pointZero.add(pointThree.sub(pointZero).multiplyScalar(0.5));
            let startPointCloned = startPoint.clone();
            let helperPoint = pointOne.add(pointTwo.sub(pointOne).multiplyScalar(0.5));
            let helperPointCloned = helperPoint.clone();
            let endPoint = startPointCloned.add(helperPointCloned.sub(startPointCloned).multiplyScalar(0.2));
            lineArray.push(this.drawLine(channelIdx, startPoint, endPoint, color, fileIndex)!);
        }


        // top four lines
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[4], channelObj.projectedPoints[5], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[5], channelObj.projectedPoints[6], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[6], channelObj.projectedPoints[7], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[7], channelObj.projectedPoints[4], color, fileIndex)!);

        // vertical lines
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[0], channelObj.projectedPoints[4], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[1], channelObj.projectedPoints[5], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[2], channelObj.projectedPoints[6], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[3], channelObj.projectedPoints[7], color, fileIndex)!);

        return lineArray;
    }

    normalize2DBoxPositions(boxPositions) {
        let normalized2DBoxPositions: number[][] = [];
        for (let i = 0; i < boxPositions.length; i++) {
            normalized2DBoxPositions.push([boxPositions[i].x / this.imageWidthOriginal, boxPositions[i].y / this.imageHeightOriginal]);
        }
        return normalized2DBoxPositions;
    }

    async update2DBoundingBox(fileIndex: number, objectIndex: number, isSelected: boolean) {
        if (objectIndex >= this.annotationObjects.contents[fileIndex].length) {
            console.log("objectIndex out of bounds");
            return;
        }
        const obj = this.annotationObjects.contents[fileIndex][objectIndex];
    
        // Single box wrapped in array
        const projectedByChannel = await this.projectBoundingBoxes([{
            x: obj.x, y: obj.y, z: obj.z,
            length: obj.length, width: obj.width, height: obj.height,
            yaw: obj.rotationYaw,
        }]);

        // The object may have been deleted while this request was in flight — bail
        // out instead of drawing a "ghost" projection for a box that no longer exists.
        if ((<any>obj).deleted) {
            return;
        }
    
        for (let channelObjectIdx in obj.channels) {
            const channelObj = obj.channels[channelObjectIdx];
            if (!channelObj.channel) continue;
            const points = projectedByChannel[channelObj.channel]?.[0] || [];
            channelObj.projectedPoints = points;
            this.removeProjectedBoundingBox(channelObj);
            // 8 = all box corners present (see calculateAndDrawLineSegments)
            if (points.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    obj.class,
                    isSelected,
                    fileIndex
                );
            }
        }

        this.drawBoundingBoxBEV(obj, isSelected, fileIndex);
    }

    drawLine(channelIdx: number, pointStart, pointEnd, color, fileIndex: number = this.labelTool.currentFrameIndex) {
        if (pointStart !== undefined && pointEnd !== undefined && isFinite(pointStart.x) && isFinite(pointStart.y) && isFinite(pointEnd.x) && isFinite(pointEnd.y)) {

            let line = this.paperArrayAll[fileIndex][channelIdx].path(
                ["M", pointStart.x, pointStart.y, "L", pointEnd.x, pointEnd.y]);
            line.attr({stroke: color, "stroke-width": 1});
            return line;
        } else {
            return undefined;
        }
    }

    async projectBoundingBoxToImage(box: AnnotationObjectParams, fileIndex: number = this.labelTool.currentFrameIndex) {
        const projectedByChannel = await this.projectBoundingBoxes([{
            x: box.x, y: box.y, z: box.z,
            length: box.length, width: box.width, height: box.height,
            yaw: box.rotationYaw,
        }]);

        // Same race as update2DBoundingBox: skip drawing if deleted while in flight.
        if ((<any>box).deleted) {
            return;
        }
    
        for (let i = 0; i < this.labelTool.cameraChannels.length; i++) {
            const channel = this.labelTool.cameraChannels[i].channel;
            box.channels[i].projectedPoints = projectedByChannel[channel]?.[0] || [];
        }
    
        // Draw lines
        for (let i = 0; i < box.channels.length; i++) {
            const channelObj = box.channels[i];
            // 8 = all box corners present (see calculateAndDrawLineSegments)
            if (channelObj.channel && channelObj.projectedPoints?.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    box.class,
                    true,
                    fileIndex
                );
            }
        }

        this.drawBoundingBoxBEV(box, true, fileIndex);
    }

    async projectPoints(points3D, channelIdx: number) {
        const labelTool3D = this.labelTool.getLabelTool3D();

        const imagePanelHeight = parseInt($("#layout_layout_resizer_top").css("top"), 10);
        const scalingFactor = this.imageHeightOriginal / imagePanelHeight;
        const projectionMatrix = this.labelTool.cameraChannels[channelIdx].projectionMatrix;

        const response = await fetch('/project_points', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points3D: points3D,
                projectionMatrix: projectionMatrix,
                scalingFactor: scalingFactor
            })
        });
        if (!response.ok) {
            throw new Error('Point projection request failed: ' + response.status);
        }
        const result = await response.json();

        // kept for compatibility with any other code reading these fields directly;
        // NOTE: unreliable when multiple channels are projected in parallel (see
        // showProjectedPoints below, which uses the returned `distances` instead).
        labelTool3D.currentPoints3D = result.points3D;
        labelTool3D.currentDistances = result.distances;

        return {
            points2D: result.points2D.map((pt: number[]) => ({ x: pt[0], y: pt[1] })),
            distances: result.distances as number[]
        };
    }

    async showProjectedPoints(points3D) {
        let labelTool3D = this.labelTool.getLabelTool3D();
        // project all channels in parallel
        const perChannelResults = await Promise.all(
            this.labelTool.cameraChannels.map((_, channelIdx) => this.projectPoints(points3D, channelIdx))
        );
        for (let channelIdx = 0; channelIdx < this.labelTool.cameraChannels.length; channelIdx++) {
            let paper = this.paperArrayAll[this.labelTool.currentFrameIndex][channelIdx];
            const { points2D, distances } = perChannelResults[channelIdx];
            // normalize this channel's distances independently (matches original
            // per-channel sequential behavior; a shared normalizeDistances() call
            // would be wrong here since channels are now computed in parallel)
            const maxDistance = distances.length > 0 ? Math.max(...distances) : 1;
            const normalizedDistances = distances.map(d => (d / maxDistance) * 255);
            for (let i = 0; i < points2D.length; i++) {
                let pt2D = points2D[i];
                let circle = paper.circle(pt2D.x, pt2D.y, 1);
                let color = labelTool3D.colorMap[Math.floor(normalizedDistances[i])];
                circle.attr("stroke", color);
                circle.attr("stroke-width", 1);
                this.circleArray.push(circle);
            }
        }
    }

    hideProjectedPoints() {
        for (let i = this.circleArray.length - 1; i >= 0; i--) {
            const circle = this.circleArray[i];
            circle.remove();
            this.circleArray.splice(i, 1);
        }
    }

    async draw2DProjection(params: any, fileIndex: number = this.labelTool.currentFrameIndex) {
        const projectedByChannel = await this.projectBoundingBoxes([{
            x: params.x, y: params.y, z: params.z,
            length: params.length, width: params.width, height: params.height,
            yaw: params.rotationYaw,
        }]);

        // Same race as update2DBoundingBox: skip drawing if deleted while in flight.
        if (params.deleted) {
            return;
        }
    
        for (let i = 0; i < params.channels.length; i++) {
            const channelObj = params.channels[i];
            if (!channelObj.channel) continue;
            channelObj.projectedPoints = projectedByChannel[channelObj.channel]?.[0] || [];
            // 8 = all box corners present (see calculateAndDrawLineSegments)
            if (channelObj.projectedPoints.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    params.class,
                    false,
                    fileIndex
                );
            }
        }

        this.drawBoundingBoxBEV(params, false, fileIndex);
    }

    async draw2DProjections() {
        const fileIndex = this.labelTool.currentFrameIndex;
        const frameContents = this.annotationObjects.contents[fileIndex];
    
        if (!frameContents || frameContents.length === 0) {
            return;
        }
    
        const boxes = frameContents.map(obj => ({
            x: obj.x,
            y: obj.y,
            z: obj.z,
            length: obj.length,
            width: obj.width,
            height: obj.height,
            yaw: obj.rotationYaw,
        }));
    
        const projectedByChannel = await this.projectBoundingBoxes(boxes);
    
        for (let i = 0; i < frameContents.length; i++) {
            const obj = frameContents[i];
            const isSelected = (this.annotationObjects.getSelectionIndex() === i);
            for (let chIdx = 0; chIdx < obj.channels.length; chIdx++) {
                const channelObj = obj.channels[chIdx];
                const channelName = channelObj.channel;
                if (!channelName)
                    continue;
    
                const points = projectedByChannel[channelName]?.[i] || [];
                channelObj.projectedPoints = points;

                this.removeProjectedBoundingBox(channelObj);

                // 8 = all box corners present (see calculateAndDrawLineSegments)
                if (points.length === 8) {
                    channelObj.lines =
                        this.calculateAndDrawLineSegments(
                            channelObj,
                            obj.class,
                            isSelected,
                            fileIndex
                        );
                }
            }

            // BEV panel: one static top-down image per frame (not per camera
            // channel), so every object's footprint is redrawn here once,
            // independent of the per-channel loop above. drawBoundingBoxBEV
            // handles removing its own previous SVG internally.
            this.drawBoundingBoxBEV(obj, isSelected, fileIndex);
        }
    }

    changeClassColorImage(bbIndex, newClass) {
        let annotation = this.annotationObjects.contents[this.labelTool.currentFrameIndex][bbIndex];
        let color = this.annotationClasses.annotationClasses[newClass].color;
        // update color in all 6 channels
        for (const channelObj of annotation["channels"]) {
            if (channelObj["lines"] !== undefined) {
                for (const line of channelObj["lines"]) {
                    // drawLine() returns undefined for any edge whose
                    // endpoint(s) fell outside the camera's calibrated
                    // FOV (see projectBoundingBoxes' null -> undefined
                    // mapping) — a partially-clipped box will have a
                    // mix of real lines and undefined slots here.
                    if (line !== undefined) {
                        line.attr({stroke: color});
                    }
                }
            }
        }

        const bev = (<any>annotation).bev;
        if (bev !== undefined && bev.lines !== undefined) {
            for (const line of bev.lines) {
                if (line !== undefined) {
                    line.attr({stroke: color});
                }
            }
        }
    }
}
export {LabelToolImage};