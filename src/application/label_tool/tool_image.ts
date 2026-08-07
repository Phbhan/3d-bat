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
        for (let lineObj in channelObj.lines) {
            if (channelObj.lines.hasOwnProperty(lineObj)) {
                let line = channelObj.lines[lineObj];
                if (line !== undefined) {
                    line.remove();
                }
            }
        }
    }

    remove2DBoundingBoxes() {
        for (let i = 0; i < this.annotationObjects.contents[this.labelTool.currentFrameIndex].length; i++) {
            for (let j = 0; j < this.annotationObjects.contents[this.labelTool.currentFrameIndex][i].channels.length; j++) {
                for (let k = 0; k < this.annotationObjects.contents[this.labelTool.currentFrameIndex][i].channels[j].lines.length; k++) {
                    let line = this.annotationObjects.contents[this.labelTool.currentFrameIndex][i].channels[j].lines[k];
                    if (line !== undefined) {
                        line.remove();
                    }
                }
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
    ): Promise<{ [channel: string]: Vector2[][] }> {
    
        const timings: { [stage: string]: number } = {};
        const totalStart = performance.now();
    
    
        // ------------------------------------
        // 1. Prepare channels
        // ------------------------------------
        let start = performance.now();

        const channels = this.labelTool.cameraChannels.map((ch, idx) => ({
            channel: ch.channel,
            zoomFactor: this.labelTool.imageScale[idx],
        }));
    
        timings["prepare_channels"] = performance.now() - start;

        // ------------------------------------
        // 2. JSON stringify
        // ------------------------------------
        start = performance.now();
    
        const requestBody = JSON.stringify({
            boxes,
            coordinateSystem: this.labelTool.coordinateSystem,
            channels,
        });
    
        timings["JSON.stringify"] = performance.now() - start;
    
        console.log(
            "Request size:",
            (requestBody.length / 1024).toFixed(2),
            "KB"
        );
    
        // ------------------------------------
        // 3. Fetch + backend
        // ------------------------------------
        start = performance.now();
    
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
    
        timings["fetch_wait"] = performance.now() - start;
    
        // ------------------------------------
        // 4. JSON parse
        // ------------------------------------
        start = performance.now();
    
        const result: {
            [channel: string]: number[][][]
        } = await response.json();
    
        timings["response.json"] = performance.now() - start;
    
        // ------------------------------------
        // 5. Convert to THREE.Vector2
        // ------------------------------------
        start = performance.now();
    
        const converted: {
            [channel: string]: Vector2[][]
        } = {};
    
    
        for (const channel in result) {
    
            converted[channel] =
                result[channel].map(
                    boxPoints =>
                        boxPoints.map(
                            pt => new THREE.Vector2(
                                pt[0],
                                pt[1]
                            )
                        )
                );
        }
    
    
        timings["Vector2 conversion"] = performance.now() - start;
        timings["TOTAL"] = performance.now() - totalStart;
    
    
        console.table(timings);
    
    
        return converted;
    }


    calculateAndDrawLineSegments(channelObj, className: string, selected: boolean, fileIndex: number = this.labelTool.currentFrameIndex) {
        let channel = channelObj.channel;
        let lineArray: RaphaelPath<"SVG" | "VML">[] = [];
        let channelIdx = Utils.getChannelIndexByName(this.labelTool.cameraChannels, channel);
        // temporary color bottom 4 lines in yellow to check if projection matrix is correct
        // uncomment line to use yellow to color bottom 4 lines
        let color;
        if (selected === true) {
            color = this.labelTool.colorSelectedObject;
        } else {
            color = this.annotationClasses.annotationClasses[className].color;
        }
        // console.log("channelObj: ", channelObj);

        // bottom four lines
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[0], channelObj.projectedPoints[1], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[1], channelObj.projectedPoints[2], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[2], channelObj.projectedPoints[3], color, fileIndex)!);
        lineArray.push(this.drawLine(channelIdx, channelObj.projectedPoints[3], channelObj.projectedPoints[0], color, fileIndex)!);

        // draw line for orientation
        let pointZero;
        let pointOne;
        let pointTwo;
        let pointThree;

        pointZero = channelObj.projectedPoints[6].clone();
        pointOne = channelObj.projectedPoints[7].clone();
        pointTwo = channelObj.projectedPoints[4].clone();
        pointThree = channelObj.projectedPoints[5].clone();


        let startPoint = pointZero.add(pointThree.sub(pointZero).multiplyScalar(0.5));
        let startPointCloned = startPoint.clone();
        let helperPoint = pointOne.add(pointTwo.sub(pointOne).multiplyScalar(0.5));
        let helperPointCloned = helperPoint.clone();
        let endPoint = startPointCloned.add(helperPointCloned.sub(startPointCloned).multiplyScalar(0.2));
        lineArray.push(this.drawLine(channelIdx, startPoint, endPoint, color, fileIndex)!);


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
            if (points.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    obj.class,
                    isSelected,
                    fileIndex
                );
            }
        }
    }

    drawLine(channelIdx: number, pointStart, pointEnd, color, fileIndex: number = this.labelTool.currentFrameIndex) {
        if (pointStart !== undefined && pointEnd !== undefined && isFinite(pointStart.x) && isFinite(pointStart.y) && isFinite(pointEnd.x) && isFinite(pointEnd.y)) {

            let line = this.paperArrayAll[fileIndex][channelIdx].path(
                ["M", pointStart.x, pointStart.y, "L", pointEnd.x, pointEnd.y]);
            line.attr("stroke", color);
            line.attr("stroke-width", 1);
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
            if (channelObj.channel && channelObj.projectedPoints?.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    box.class,
                    true,
                    fileIndex
                );
            }
        }
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
            if (channelObj.projectedPoints.length === 8) {
                channelObj.lines = this.calculateAndDrawLineSegments(
                    channelObj,
                    params.class,
                    false,
                    fileIndex
                );
            }
        }
    }

    async draw2DProjections() {
        const timings: { [stage: string]: number } = {};
        const totalStart = performance.now();
        const fileIndex = this.labelTool.currentFrameIndex;
        const frameContents = this.annotationObjects.contents[fileIndex];
    
        if (!frameContents || frameContents.length === 0) {
            return;
        }
    
        // --------------------------------------------------
        // 1. Prepare boxes
        // --------------------------------------------------
        let start = performance.now();
        const boxes = frameContents.map(obj => ({
            x: obj.x,
            y: obj.y,
            z: obj.z,
            length: obj.length,
            width: obj.width,
            height: obj.height,
            yaw: obj.rotationYaw,
        }));
    
        timings["prepare_boxes"] = performance.now() - start;

        // --------------------------------------------------
        // 2. Projection request
        // --------------------------------------------------
        start = performance.now();
        const projectedByChannel = await this.projectBoundingBoxes(boxes);
        timings["project_request"] = performance.now() - start;

        // --------------------------------------------------
        // 3. Process objects
        // --------------------------------------------------
        let assignTime = 0;
        let removeTime = 0;
        let drawTime = 0;
    
        for (let i = 0; i < frameContents.length; i++) {
            const obj = frameContents[i];
            for (let chIdx = 0; chIdx < obj.channels.length; chIdx++) {
                const channelObj = obj.channels[chIdx];
                const channelName = channelObj.channel;
                if (!channelName)
                    continue;
    
                // -----------------------------
                // assign points
                // -----------------------------
                start = performance.now();
                const points = projectedByChannel[channelName]?.[i] || [];
                channelObj.projectedPoints = points;
                assignTime += performance.now() - start;

                // -----------------------------
                // remove old SVG
                // -----------------------------
                start = performance.now();
                this.removeProjectedBoundingBox(channelObj);
                removeTime += performance.now() - start;

                // -----------------------------
                // draw SVG
                // -----------------------------
                if (points.length === 8) {
    
                    start = performance.now();
                    const isSelected = (this.annotationObjects.getSelectionIndex() === i);
                    channelObj.lines =
                        this.calculateAndDrawLineSegments(
                            channelObj,
                            obj.class,
                            isSelected,
                            fileIndex
                        );
    
                    drawTime += performance.now() - start;
                }
            }
        }
    
    
        timings["assign_points"] = assignTime;
        timings["remove_lines"] = removeTime;
        timings["draw_lines"] = drawTime;
    
    
        timings["TOTAL"] = performance.now() - totalStart;
    
    
        console.table(timings);
    }

    changeClassColorImage(bbIndex, newClass) {
        let annotation = this.annotationObjects.contents[this.labelTool.currentFrameIndex][bbIndex];
        let color = this.annotationClasses.annotationClasses[newClass].color;
        // update color in all 6 channels
        for (let i = 0; i < annotation["channels"].length; i++) {
            if (annotation["channels"][i]["lines"] !== undefined && annotation["channels"][i]["lines"][0] !== undefined) {
                for (let lineObj in annotation["channels"][i]["lines"]) {
                    if (annotation["channels"][i]["lines"].hasOwnProperty(lineObj)) {
                        const line = annotation["channels"][i]["lines"][lineObj];
                        line.attr({stroke: color});
                    }
                }
            }
        }
    }
}
export {LabelToolImage};