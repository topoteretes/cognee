import { Color } from "three";
import { Text } from "troika-three-text";

const LABEL_FONT_SIZE = 14;

export default function createLabel(text = "", fontSize = LABEL_FONT_SIZE): Text {
  const label = new Text();
  label.text = text;
  label.fontSize = fontSize;
  label.color = new Color("#ffffff");
  label.strokeColor = new Color("#ffffff");
  label.outlineWidth = 2;
  label.outlineColor = new Color("#000000");
  label.outlineOpacity = 0.5;
  label.anchorX = "center";
  label.anchorY = "middle";
  label.visible = true;
  label.frustumCulled = false;
  label.renderOrder = 5;
  label.maxWidth = 200;
  label.sync();

  return label;
}
