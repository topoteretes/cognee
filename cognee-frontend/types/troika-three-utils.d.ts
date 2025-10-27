// /* eslint-disable @typescript-eslint/no-explicit-any */
declare module "troika-three-utils";

// import type { Color, Material, Object3D, Object3DEventMap } from "three";

// export class BatchedText {
//   constructor(...args: any[]);

//   add(...args: any[]): void;

//   addText(...args: any[]): void;

//   copy(...args: any[]): void;

//   createDerivedMaterial(...args: any[]): void;

//   dispose(...args: any[]): void;

//   hasOutline(...args: any[]): void;

//   remove(...args: any[]): void;

//   removeText(...args: any[]): void;

//   sync(...args: any[]): void;

//   updateBounds(...args: any[]): void;

//   updateMatrixWorld(...args: any[]): void;

//   static DEFAULT_MATRIX_AUTO_UPDATE: boolean;

//   static DEFAULT_MATRIX_WORLD_AUTO_UPDATE: boolean;
// }

// export class GlyphsGeometry {
//   constructor(...args: any[]);

//   applyClipRect(...args: any[]): void;

//   computeBoundingBox(...args: any[]): void;

//   computeBoundingSphere(...args: any[]): void;

//   updateAttributeData(...args: any[]): void;

//   updateGlyphs(...args: any[]): void;
// }

// export class Text extends Object3D<Object3DEventMap> {
//   public text: string;
//   public fontSize: number;
//   public color: Color;
//   public anchorX;
//   public anchorY;
//   public font: string;
//   public material: Material;

//   constructor(...args: any[]): Object3D<Object3DEventMap>;

//   clone(...args: any[]): Object3D<Object3DEventMap>;

//   copy(...args: any[]): Object3D<Object3DEventMap>;

//   createDerivedMaterial(...args: any[]): void;

//   dispose(...args: any[]): void;

//   hasOutline(...args: any[]): void;

//   localPositionToTextCoords(...args: any[]): void;

//   onBeforeRender(...args: any[]): void;

//   raycast(...args: any[]): void;

//   sync(...args: any[]): void;

//   worldPositionToTextCoords(...args: any[]): void;

//   static DEFAULT_MATRIX_AUTO_UPDATE: boolean;

//   static DEFAULT_MATRIX_WORLD_AUTO_UPDATE: boolean;
// }

// export function configureTextBuilder(config: any): void;

// export function createTextDerivedMaterial(baseMaterial: any): any;

// export function dumpSDFTextures(): void;

// export function fontResolverWorkerModule(...args: any[]): any;

// export function getCaretAtPoint(textRenderInfo: any, x: any, y: any): any;

// export function getSelectionRects(
//   textRenderInfo: any,
//   start: any,
//   end: any
// ): any;

// export function getTextRenderInfo(args: any, callback: any): any;

// export function preloadFont(
//   { font, characters, sdfGlyphSize }: any,
//   callback: any
// ): void;

// export function typesetterWorkerModule(...args: any[]): any;

// export namespace BatchedText {
//   namespace DEFAULT_UP {
//     const isVector3: boolean;

//     const x: number;

//     const y: number;

//     const z: number;

//     function add(...args: any[]): void;

//     function addScalar(...args: any[]): void;

//     function addScaledVector(...args: any[]): void;

//     function addVectors(...args: any[]): void;

//     function angleTo(...args: any[]): void;

//     function applyAxisAngle(...args: any[]): void;

//     function applyEuler(...args: any[]): void;

//     function applyMatrix3(...args: any[]): void;

//     function applyMatrix4(...args: any[]): void;

//     function applyNormalMatrix(...args: any[]): void;

//     function applyQuaternion(...args: any[]): void;

//     function ceil(...args: any[]): void;

//     function clamp(...args: any[]): void;

//     function clampLength(...args: any[]): void;

//     function clampScalar(...args: any[]): void;

//     function clone(...args: any[]): void;

//     function copy(...args: any[]): void;

//     function cross(...args: any[]): void;

//     function crossVectors(...args: any[]): void;

//     function distanceTo(...args: any[]): void;

//     function distanceToSquared(...args: any[]): void;

//     function divide(...args: any[]): void;

//     function divideScalar(...args: any[]): void;

//     function dot(...args: any[]): void;

//     function equals(...args: any[]): void;

//     function floor(...args: any[]): void;

//     function fromArray(...args: any[]): void;

//     function fromBufferAttribute(...args: any[]): void;

//     function getComponent(...args: any[]): void;

//     function length(...args: any[]): void;

//     function lengthSq(...args: any[]): void;

//     function lerp(...args: any[]): void;

//     function lerpVectors(...args: any[]): void;

//     function manhattanDistanceTo(...args: any[]): void;

//     function manhattanLength(...args: any[]): void;

//     function max(...args: any[]): void;

//     function min(...args: any[]): void;

//     function multiply(...args: any[]): void;

//     function multiplyScalar(...args: any[]): void;

//     function multiplyVectors(...args: any[]): void;

//     function negate(...args: any[]): void;

//     function normalize(...args: any[]): void;

//     function project(...args: any[]): void;

//     function projectOnPlane(...args: any[]): void;

//     function projectOnVector(...args: any[]): void;

//     function random(...args: any[]): void;

//     function randomDirection(...args: any[]): void;

//     function reflect(...args: any[]): void;

//     function round(...args: any[]): void;

//     function roundToZero(...args: any[]): void;

//     function set(...args: any[]): void;

//     function setComponent(...args: any[]): void;

//     function setFromColor(...args: any[]): void;

//     function setFromCylindrical(...args: any[]): void;

//     function setFromCylindricalCoords(...args: any[]): void;

//     function setFromEuler(...args: any[]): void;

//     function setFromMatrix3Column(...args: any[]): void;

//     function setFromMatrixColumn(...args: any[]): void;

//     function setFromMatrixPosition(...args: any[]): void;

//     function setFromMatrixScale(...args: any[]): void;

//     function setFromSpherical(...args: any[]): void;

//     function setFromSphericalCoords(...args: any[]): void;

//     function setLength(...args: any[]): void;

//     function setScalar(...args: any[]): void;

//     function setX(...args: any[]): void;

//     function setY(...args: any[]): void;

//     function setZ(...args: any[]): void;

//     function sub(...args: any[]): void;

//     function subScalar(...args: any[]): void;

//     function subVectors(...args: any[]): void;

//     function toArray(...args: any[]): void;

//     function transformDirection(...args: any[]): void;

//     function unproject(...args: any[]): void;
//   }
// }

// export namespace Text {
//   namespace DEFAULT_UP {
//     const isVector3: boolean;

//     const x: number;

//     const y: number;

//     const z: number;

//     function add(...args: any[]): void;

//     function addScalar(...args: any[]): void;

//     function addScaledVector(...args: any[]): void;

//     function addVectors(...args: any[]): void;

//     function angleTo(...args: any[]): void;

//     function applyAxisAngle(...args: any[]): void;

//     function applyEuler(...args: any[]): void;

//     function applyMatrix3(...args: any[]): void;

//     function applyMatrix4(...args: any[]): void;

//     function applyNormalMatrix(...args: any[]): void;

//     function applyQuaternion(...args: any[]): void;

//     function ceil(...args: any[]): void;

//     function clamp(...args: any[]): void;

//     function clampLength(...args: any[]): void;

//     function clampScalar(...args: any[]): void;

//     function clone(...args: any[]): void;

//     function copy(...args: any[]): void;

//     function cross(...args: any[]): void;

//     function crossVectors(...args: any[]): void;

//     function distanceTo(...args: any[]): void;

//     function distanceToSquared(...args: any[]): void;

//     function divide(...args: any[]): void;

//     function divideScalar(...args: any[]): void;

//     function dot(...args: any[]): void;

//     function equals(...args: any[]): void;

//     function floor(...args: any[]): void;

//     function fromArray(...args: any[]): void;

//     function fromBufferAttribute(...args: any[]): void;

//     function getComponent(...args: any[]): void;

//     function length(...args: any[]): void;

//     function lengthSq(...args: any[]): void;

//     function lerp(...args: any[]): void;

//     function lerpVectors(...args: any[]): void;

//     function manhattanDistanceTo(...args: any[]): void;

//     function manhattanLength(...args: any[]): void;

//     function max(...args: any[]): void;

//     function min(...args: any[]): void;

//     function multiply(...args: any[]): void;

//     function multiplyScalar(...args: any[]): void;

//     function multiplyVectors(...args: any[]): void;

//     function negate(...args: any[]): void;

//     function normalize(...args: any[]): void;

//     function project(...args: any[]): void;

//     function projectOnPlane(...args: any[]): void;

//     function projectOnVector(...args: any[]): void;

//     function random(...args: any[]): void;

//     function randomDirection(...args: any[]): void;

//     function reflect(...args: any[]): void;

//     function round(...args: any[]): void;

//     function roundToZero(...args: any[]): void;

//     function set(...args: any[]): void;

//     function setComponent(...args: any[]): void;

//     function setFromColor(...args: any[]): void;

//     function setFromCylindrical(...args: any[]): void;

//     function setFromCylindricalCoords(...args: any[]): void;

//     function setFromEuler(...args: any[]): void;

//     function setFromMatrix3Column(...args: any[]): void;

//     function setFromMatrixColumn(...args: any[]): void;

//     function setFromMatrixPosition(...args: any[]): void;

//     function setFromMatrixScale(...args: any[]): void;

//     function setFromSpherical(...args: any[]): void;

//     function setFromSphericalCoords(...args: any[]): void;

//     function setLength(...args: any[]): void;

//     function setScalar(...args: any[]): void;

//     function setX(...args: any[]): void;

//     function setY(...args: any[]): void;

//     function setZ(...args: any[]): void;

//     function sub(...args: any[]): void;

//     function subScalar(...args: any[]): void;

//     function subVectors(...args: any[]): void;

//     function toArray(...args: any[]): void;

//     function transformDirection(...args: any[]): void;

//     function unproject(...args: any[]): void;
//   }
// }

// export namespace fontResolverWorkerModule {
//   const workerModuleData: {
//     dependencies: {
//       dependencies: any;
//       getTransferables: any;
//       id: string;
//       init: string;
//       isWorkerModule: boolean;
//       name: string;
//     }[];
//     getTransferables: any;
//     id: string;
//     init: string;
//     isWorkerModule: boolean;
//     name: string;
//   };

//   function onMainThread(...args: any[]): any;
// }

// export namespace typesetterWorkerModule {
//   const workerModuleData: {
//     dependencies: {
//       dependencies: any;
//       getTransferables: any;
//       id: string;
//       init: string;
//       isWorkerModule: boolean;
//       name: string;
//     }[];
//     getTransferables: any;
//     id: string;
//     init: string;
//     isWorkerModule: boolean;
//     name: string;
//   };

//   function onMainThread(...args: any[]): any;
// }
