import {
  FaceSmileIcon,
  FaceFrownIcon,
  FireIcon,
  BoltIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  MinusCircleIcon,
} from "@heroicons/react/24/outline";

export const emotionMap = {
  happy: {
    icon: FaceSmileIcon,
    color: "text-green-500",
  },
  sad: {
    icon: FaceFrownIcon,
    color: "text-blue-500",
  },
  angry: {
    icon: FireIcon,
    color: "text-red-500",
  },
  surprise: {
    icon: BoltIcon,
    color: "text-yellow-500",
  },
  fear: {
    icon: ExclamationTriangleIcon,
    color: "text-orange-500",
  },
  disgust: {
    icon: XCircleIcon,
    color: "text-slate-500",
  },
  neutral: {
    icon: MinusCircleIcon,
    color: "text-gray-500",
  },
};