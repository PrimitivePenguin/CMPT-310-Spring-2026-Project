export async function predictEmotion(file) {
  if (!file) {
    throw new Error("No file provided");
  }

  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error("Invalid server response");
  }

  if (!response.ok) {
    throw new Error(data?.error || "Prediction failed");
  }

  return data;
}