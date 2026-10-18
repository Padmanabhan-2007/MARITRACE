"""Generate a simulated incident PDF when reportlab is installed."""
def create_report(path,scene,ranked,metrics=None):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    c=canvas.Canvas(str(path),pagesize=letter); w,h=letter
    c.setFont('Helvetica-Bold',20); c.drawString(48,h-55,'MARITRACE Incident Report')
    c.setFont('Helvetica-Bold',14); c.setFillColorRGB(.75,.1,.1); c.drawString(48,h-82,'SIMULATED DATA — NOT FOR REAL-WORLD ATTRIBUTION')
    c.setFillColorRGB(0,0,0); c.setFont('Helvetica',10); c.drawString(48,h-112,f"Scene timestamp: {scene['timestamp']}"); c.drawString(48,h-130,f"Spill location: {scene['lat']:.4f}, {scene['lon']:.4f}")
    c.drawString(48,h-160,'Ranked heuristic suspects:')
    for i,row in ranked.head(8).iterrows(): c.drawString(55,h-180-i*17,f"{i+1}. {row.get('name',row.mmsi)}  score={row.score:.3f}")
    if metrics:
        c.drawString(48,140,f"Synthetic test pixel accuracy: {metrics['pixel_accuracy']:.3f}; oil IoU: {metrics['iou'][1]:.3f}")
    c.drawString(48,100,'Synthetic generator metrics do not demonstrate performance on real SAR imagery.'); c.drawString(48,82,'Wind/current are simulated; vessel ranking is heuristic and requires independent investigation.')
    c.save()
