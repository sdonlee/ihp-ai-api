from ai.aiapi import EasyNeg

easyneg = EasyNeg(video_path = video_, DEBUG=True)
easyneg.analyze_neg()

print(f"{easyneg.scr_drk:.2f}, {easyneg.scr_mtl:.2f}")
print([f"{key_}: {val_:.2f}," for key_, val_ in zip(easyneg.time.keys(), easyneg.time.values())])