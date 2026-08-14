import {Utils} from "../../util/utils";
import {LabelTool} from "../tool_main";
import {AnnotationsLoader} from "./loader";

// Everything in a frame file except the "labels" array. Stored per-frame so it can
// be round-tripped unchanged on save (name/cam_pos/timestamp/weather aren't derived
// from any annotation object, they're just frame-level metadata from the file).
interface FrameMeta {
    name: string;
    cam_pos: string;
    timestamp: number;
    index: number;
    weather: string;
}

export class dataLoader implements AnnotationsLoader {
    getFilename = (labelTool: LabelTool, i: number) => labelTool.annotationFileNames[i];

    // Used when a frame has no annotation file yet (new/empty frame).
    // The "visibility" attribute's key in annotationObj.attributes must match whatever
    // name the config gives it (e.g. "Visibility"), since that's the key
    // createDropDownAttribute() in tool_3d.ts uses to read/write the GUI dropdown.
    // Hardcoding "visibility" here caused edits to silently not persist, because the
    // dropdown was writing to a differently-cased key.
    private getVisibilityAttributeKey(labelTool: LabelTool, objectClassIdx: number): string {
        const datasetConfig = labelTool.config.datasets[labelTool.datasetArray.indexOf(labelTool.currentDataset)];
        const classConfig = datasetConfig.classes[objectClassIdx];
        const attrConfig = classConfig?.attributes?.find((a: any) => a.name?.toLowerCase() === "visibility");
        return attrConfig ? attrConfig.name : "visibility";
    }

    private defaultFrameMeta(labelTool: LabelTool, fileIndex: number): FrameMeta {
        const datasetConfig = labelTool.config.datasets[labelTool.datasetArray.indexOf(labelTool.currentDataset)];
        return {
            name: fileIndex.toString().padStart(6, "0"),
            cam_pos: datasetConfig.camera_channels?.[0]?.channel ?? "",
            timestamp: fileIndex,
            index: fileIndex,
            weather: datasetConfig.weather_type?.[0] ?? ""
        };
    }

    loadAnnotations = (frameObject: any, fileIndex: number, labelTool: LabelTool) => {
        if (!frameObject || !Array.isArray(frameObject.labels)) {
            console.warn(`No annotation data for frame ${fileIndex} — treating as empty frame.`);
            labelTool.frameProperties[fileIndex] = this.defaultFrameMeta(labelTool, fileIndex);
            return;
        }

        // Keep every field except "labels" so the frame metadata round-trips exactly
        // as loaded, instead of being recomputed/guessed on save.
        const {labels, ...frameMeta} = frameObject;
        labelTool.frameProperties[fileIndex] = {
            name: frameMeta.name ?? fileIndex.toString().padStart(6, "0"),
            cam_pos: frameMeta.cam_pos ?? "",
            timestamp: frameMeta.timestamp ?? fileIndex,
            index: frameMeta.index ?? fileIndex,
            weather: frameMeta.weather ?? ""
        };

        labels.forEach((label: any, idx: number) => {
            const box3d = label.box3d;
            const dimension = box3d.dimension;
            const location = box3d.location;
            const orientation = box3d.orientation;

            let params = labelTool.annotationObjects.getDefaultObject();

            // category already matches the class names in config.json (CAR, TRUCK, ...)
            params.class = label.category;
            params.original.class = label.category;

            // This format stores plain Euler angles directly — no quaternion decomposition needed.
            params.rotationYaw = orientation.rotationYaw;
            params.original.rotationYaw = orientation.rotationYaw;
            params.rotationPitch = orientation.rotationPitch;
            params.original.rotationPitch = orientation.rotationPitch;
            params.rotationRoll = orientation.rotationRoll;
            params.original.rotationRoll = orientation.rotationRoll;

            // NOTE: this format has no persisted track id (label.id is just this
            // object's position within the frame, dropped on save and reassigned
            // sequentially). We still need *some* unique key internally for GUI
            // folders / transform controls, so we use the in-frame index as trackId.
            // Because it isn't stable across frames, cross-frame tracking/interpolation
            // by trackId will not reliably follow the same object between frames.
            params.trackId = idx.toString();
            params.original.trackId = idx.toString();

            params.x = location.x;
            params.original.x = location.x;
            params.y = location.y;
            params.original.y = location.y;
            params.z = location.z;
            params.original.z = location.z;

            params.length = Math.max(dimension.length, 0.0001);
            params.original.length = params.length;
            params.width = Math.max(dimension.width, 0.0001);
            params.original.width = params.width;
            params.height = Math.max(dimension.height, 0.0001);
            params.original.height = params.height;

            params.fileIndex = fileIndex;

            let objectClassIdx = labelTool.annotationClasses.getIndexByObjectClass(label.category);
            let defaultAttributes = labelTool.annotationObjects.getDefaultAttributesByClassIdx(objectClassIdx);

            // "visibility" attribute
            const visibilityKey = this.getVisibilityAttributeKey(labelTool, objectClassIdx);
            params.attributes = {
                ...defaultAttributes,
                [visibilityKey]: box3d.visibility ?? defaultAttributes[visibilityKey]
            };

            labelTool.annotationObjects.set(labelTool.annotationObjects.__insertIndex, params);
            labelTool.annotationObjects.__insertIndex++;
        });
        labelTool.annotationObjects.__insertIndex = 0;
    }

    createAnnotationFiles = (labelTool: LabelTool) => {
        const annotationFiles: string[] = [];
        for (let j = 0; j < labelTool.numFrames; j++) {
            const labels: any[] = [];
            let nextId = 0;

            for (let i = 0; i < labelTool.annotationObjects.contents[j].length; i++) {
                if (labelTool.annotationObjects.contents[j][i] !== undefined && labelTool.cubeArray[j][i] !== undefined) {
                    const annotationObj = labelTool.annotationObjects.contents[j][i];
                    const cube = labelTool.cubeArray[j][i];

                    let objectClassIdx = Utils.getIndexByClass(
                        labelTool.config.datasets[labelTool.datasetArray.indexOf(labelTool.currentDataset)].classes,
                        annotationObj["class"]
                    );
                    let defaultAttributes = labelTool.annotationObjects.getDefaultAttributesByClassIdx(objectClassIdx);
                    const visibilityKey = this.getVisibilityAttributeKey(labelTool, objectClassIdx);

                    let visibility = defaultAttributes[visibilityKey];
                    if (annotationObj["attributes"] && annotationObj["attributes"][visibilityKey] !== undefined) {
                        visibility = annotationObj["attributes"][visibilityKey];
                    }

                    // scale.x = length (forward axis), scale.y = width, scale.z = height
                    // — same convention used throughout tool_3d.ts (cubeLength/cubeWidth/cubeHeight)
                    labels.push({
                        id: nextId++,
                        category: annotationObj["class"],
                        box3d: {
                            dimension: {
                                width: cube.scale.y,
                                length: cube.scale.x,
                                height: cube.scale.z
                            },
                            location: {
                                x: cube.position.x,
                                y: cube.position.y,
                                z: cube.position.z
                            },
                            orientation: {
                                rotationYaw: cube.rotation.z,
                                rotationPitch: cube.rotation.y,
                                rotationRoll: cube.rotation.x
                            },
                            visibility: visibility
                        }
                    });
                }
            }

            const frameMeta: FrameMeta = labelTool.frameProperties[j] ?? this.defaultFrameMeta(labelTool, j);

            const frameJSON = {
                name: frameMeta.name,
                cam_pos: frameMeta.cam_pos,
                timestamp: frameMeta.timestamp,
                index: frameMeta.index,
                weather: frameMeta.weather,
                labels: labels
            };
            annotationFiles.push(JSON.stringify(frameJSON));
        }
        return annotationFiles;
    }
}