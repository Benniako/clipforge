/**
 * Multi-Platform Phone Canvas Collision Engine & Safe Zone Registry
 * Milestone M4: Requirements R4 & R5 (PROJECT.md § 5 & Explorer UI Handoff § 2.3)
 *
 * Implements 2D Axis-Aligned Bounding Box (AABB) intersection detection,
 * exclusion zones for TikTok, YouTube Shorts, and Instagram Reels,
 * and deterministic auto-nudge repositioning algorithms.
 */

import type { Rect } from "./types";

// Re-export Rect for convenience
export type { Rect };

export type PlatformType = "tiktok" | "shorts" | "reels";

export type ZoneType = "action_rail" | "bottom_meta" | "top_nav" | "bottom_bar";

export type CollisionSeverity = "critical" | "warning" | "info";

export interface PlatformUIZone {
  id: string;
  platform: PlatformType;
  name: string;
  displayName: string;
  type: ZoneType;
  rect: Rect; // Normalized coordinates [0.0, 1.0] relative to 9:16 frame
  severity: CollisionSeverity;
  description?: string;
}

// Alias for backwards-compatibility
export type PlatformZone = PlatformUIZone;

export interface CollisionResult {
  zoneId: string;
  zoneName: string;
  displayName: string;
  zoneType: ZoneType;
  platform: PlatformType;
  severity: CollisionSeverity;
  overlapArea: number;
  overlapRect: Rect;
  zoneRect: Rect;
  elementRect: Rect;
  message: string;
}

export interface MagneticSnapResult {
  time: number;
  snapped: boolean;
  snapPoint?: number;
  delta?: number;
}

// ============================================================================
// Platform Exclusion Zones Registry (Normalized [0.0, 1.0] in 9:16 space)
// ============================================================================

export const PLATFORM_UI_ZONES: Record<PlatformType, PlatformUIZone[]> = {
  tiktok: [
    {
      id: "tt_top",
      platform: "tiktok",
      name: "top_tabs",
      displayName: "TikTok Top Header & Tabs",
      type: "top_nav",
      rect: { x: 0.15, y: 0.01, w: 0.70, h: 0.09 },
      severity: "warning",
      description: "Live, Following, For You tabs & Search bar",
    },
    {
      id: "tt_rail",
      platform: "tiktok",
      name: "right_rail_actions",
      displayName: "TikTok Action Rail",
      type: "action_rail",
      rect: { x: 0.82, y: 0.42, w: 0.17, h: 0.46 },
      severity: "critical",
      description: "Like, Comment, Favorite, Share buttons & Audio Disc",
    },
    {
      id: "tt_meta",
      platform: "tiktok",
      name: "bottom_creator_sound",
      displayName: "TikTok Creator & Sound Metadata",
      type: "bottom_meta",
      rect: { x: 0.04, y: 0.76, w: 0.78, h: 0.20 },
      severity: "critical",
      description: "Creator handle, caption description, hashtags & music ticker",
    },
    {
      id: "tt_bar",
      platform: "tiktok",
      name: "bottom_nav_bar",
      displayName: "TikTok Bottom Navigation Bar",
      type: "bottom_bar",
      rect: { x: 0.0, y: 0.96, w: 1.0, h: 0.04 },
      severity: "info",
      description: "Home, Friends, Create, Inbox, Profile bar",
    },
  ],
  shorts: [
    {
      id: "ys_top",
      platform: "shorts",
      name: "top_search_controls",
      displayName: "Shorts Top Navigation & Search",
      type: "top_nav",
      rect: { x: 0.04, y: 0.01, w: 0.92, h: 0.08 },
      severity: "warning",
      description: "Camera icon, search bar, and 3-dot overflow menu",
    },
    {
      id: "ys_rail",
      platform: "shorts",
      name: "right_rail_engagement",
      displayName: "Shorts Engagement Rail",
      type: "action_rail",
      rect: { x: 0.82, y: 0.46, w: 0.17, h: 0.44 },
      severity: "critical",
      description: "Like, Dislike, Comments, Share, Remix buttons & Sound icon",
    },
    {
      id: "ys_meta",
      platform: "shorts",
      name: "bottom_channel_title",
      displayName: "Shorts Channel & Title Metadata",
      type: "bottom_meta",
      rect: { x: 0.04, y: 0.78, w: 0.78, h: 0.18 },
      severity: "critical",
      description: "Channel avatar, Subscribe button, Title snippet & Sound attribution",
    },
    {
      id: "ys_bar",
      platform: "shorts",
      name: "bottom_nav_bar",
      displayName: "Shorts Bottom Navigation Bar",
      type: "bottom_bar",
      rect: { x: 0.0, y: 0.96, w: 1.0, h: 0.04 },
      severity: "info",
      description: "YouTube standard bottom navigation bar",
    },
  ],
  reels: [
    {
      id: "ig_top",
      platform: "reels",
      name: "top_camera_nav",
      displayName: "Reels Top Header & Camera",
      type: "top_nav",
      rect: { x: 0.04, y: 0.01, w: 0.92, h: 0.09 },
      severity: "warning",
      description: "Reels title header and camera button",
    },
    {
      id: "ig_rail",
      platform: "reels",
      name: "right_rail_reels_actions",
      displayName: "Reels Action Rail",
      type: "action_rail",
      rect: { x: 0.82, y: 0.48, w: 0.17, h: 0.42 },
      severity: "critical",
      description: "Heart, Comment, Share (Paper Airplane), Menu & Audio square",
    },
    {
      id: "ig_meta",
      platform: "reels",
      name: "bottom_reels_audio_bar",
      displayName: "Reels Profile & Audio Bar",
      type: "bottom_meta",
      rect: { x: 0.04, y: 0.78, w: 0.78, h: 0.18 },
      severity: "critical",
      description: "Profile avatar, Follow pill, Caption & Audio track marquee",
    },
    {
      id: "ig_bar",
      platform: "reels",
      name: "bottom_nav_bar",
      displayName: "Instagram Bottom Navigation Bar",
      type: "bottom_bar",
      rect: { x: 0.0, y: 0.96, w: 1.0, h: 0.04 },
      severity: "info",
      description: "Instagram standard tab bar",
    },
  ],
};

// Aliased dictionary matching tests
export const PLATFORM_OVERLAYS = PLATFORM_UI_ZONES;

// ============================================================================
// AABB Collision Detection Algorithms
// ============================================================================

/**
 * 2D Axis-Aligned Bounding Box (AABB) intersection check.
 * Returns true if box a and box b overlap by a non-zero area.
 */
export function checkCollision(a: Rect, b: Rect): boolean {
  return (
    a.x < b.x + b.w &&
    a.x + a.w > b.x &&
    a.y < b.y + b.h &&
    a.y + a.h > b.y
  );
}

/**
 * Functional alias for checkCollision.
 */
export function boxesIntersect(a: Rect, b: Rect): boolean {
  return checkCollision(a, b);
}

/**
 * Calculates the 2D intersection area of two rectangles in normalized coordinate space.
 */
export function intersectionArea(a: Rect, b: Rect): number {
  const xOverlap = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const yOverlap = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return xOverlap * yOverlap;
}

/**
 * Returns the intersection rectangle if overlapping, or null otherwise.
 */
export function getIntersectionRect(a: Rect, b: Rect): Rect | null {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  if (x2 <= x1 || y2 <= y1) return null;
  return { x: x1, y: y1, w: x2 - x1, h: y2 - y1 };
}

/**
 * Detects all platform UI zones colliding with the target bounding box.
 * Returns list of matching PlatformUIZone objects.
 */
export function detectPlatformCollisions(target: Rect, platform: PlatformType): PlatformUIZone[] {
  const zones = PLATFORM_UI_ZONES[platform] || [];
  return zones.filter((zone) => checkCollision(target, zone.rect));
}

/**
 * Comprehensive platform collision check returning detailed CollisionResult objects.
 */
export function checkPlatformCollisions(
  elementRect: Rect,
  platform: PlatformType,
  elementLabel = "Element"
): CollisionResult[] {
  const zones = PLATFORM_UI_ZONES[platform] || [];
  const results: CollisionResult[] = [];

  for (const zone of zones) {
    if (checkCollision(elementRect, zone.rect)) {
      const overlap = getIntersectionRect(elementRect, zone.rect) || { x: 0, y: 0, w: 0, h: 0 };
      const area = overlap.w * overlap.h;

      results.push({
        zoneId: zone.id,
        zoneName: zone.name,
        displayName: zone.displayName,
        zoneType: zone.type,
        platform,
        severity: zone.severity,
        overlapArea: area,
        overlapRect: overlap,
        zoneRect: zone.rect,
        elementRect,
        message: `⚠️ Collision: ${elementLabel} overlaps ${zone.displayName} (${zone.name})`,
      });
    }
  }

  // Sort critical collisions first, then by overlap area descending
  return results.sort((a, b) => {
    if (a.severity === "critical" && b.severity !== "critical") return -1;
    if (b.severity === "critical" && a.severity !== "critical") return 1;
    return b.overlapArea - a.overlapArea;
  });
}

// ============================================================================
// Auto-Nudge Safe Repositioning Algorithms
// ============================================================================

/**
 * Calculates auto-nudge repositioning to clear platform collisions.
 * Preserves element width and height, moving it into the nearest safe area.
 */
export function calculateAutoNudge(target: Rect, collisions: PlatformZone[]): Rect {
  if (!collisions || collisions.length === 0) return { ...target };

  let nudged = { ...target };

  for (const zone of collisions) {
    // If collision is in right rail, nudge leftward away from the rail
    if ((zone.name.includes("right") || zone.type === "action_rail") && nudged.x + nudged.w > zone.rect.x) {
      nudged.x = Math.max(0.04, zone.rect.x - nudged.w - 0.02);
    }

    // If collision is in the bottom area, nudge upward above the metadata
    if ((zone.name.includes("bottom") || zone.type === "bottom_meta" || zone.type === "bottom_bar") && nudged.y + nudged.h > zone.rect.y) {
      nudged.y = Math.max(0.06, zone.rect.y - nudged.h - 0.02);
    }

    // If collision is in top area, nudge downward
    if ((zone.name.includes("top") || zone.type === "top_nav") && nudged.y < zone.rect.y + zone.rect.h) {
      nudged.y = Math.min(0.90 - nudged.h, zone.rect.y + zone.rect.h + 0.02);
    }
  }

  // Final bounds clamping [0.02, 0.98]
  nudged.x = Math.max(0.02, Math.min(1.0 - nudged.w - 0.02, nudged.x));
  nudged.y = Math.max(0.02, Math.min(1.0 - nudged.h - 0.02, nudged.y));

  return nudged;
}

/**
 * High-level auto-nudge function that calculates the optimal collision-free position
 * for captions or facecam overlays for the given platform.
 */
export function autoNudgeSafePosition(
  elementRect: Rect,
  platform: PlatformType,
  options?: {
    elementRole?: "caption" | "facecam";
    preferredDirection?: "up" | "left" | "down" | "center";
  }
): Rect {
  const collisions = detectPlatformCollisions(elementRect, platform);
  if (collisions.length === 0) {
    return { ...elementRect };
  }

  let nudged = calculateAutoNudge(elementRect, collisions);

  // If still colliding, fallback to guaranteed platform safe zone
  const remainingCollisions = detectPlatformCollisions(nudged, platform);
  if (remainingCollisions.length > 0) {
    const isCaption = options?.elementRole === "caption" || elementRect.w > 0.4;
    if (isCaption) {
      // Safe vertical zone for captions in 9:16 is between top nav (y=0.10) and bottom meta (y=0.74)
      const safeW = Math.min(0.72, elementRect.w);
      const safeH = Math.min(0.18, elementRect.h);
      // Center fallback captions within the platform safe horizontal range [0.04, 0.80]
      const safeMinX = 0.04;
      const safeMaxX = 0.80;
      const safeX = safeMinX + (safeMaxX - safeMinX - safeW) / 2;
      const safeY = Math.min(0.56, Math.max(0.12, 0.74 - safeH));
      nudged = {
        x: Math.max(safeMinX, Math.min(safeMaxX - safeW, safeX)),
        y: safeY,
        w: safeW,
        h: safeH,
      };
    } else {
      // Facecam safe placement: top-left corner below top nav
      const safeW = Math.min(0.35, elementRect.w);
      const safeH = Math.min(0.35, elementRect.h);
      nudged = {
        x: 0.04,
        y: 0.12,
        w: safeW,
        h: safeH,
      };
    }
  }

  // Strict frame bounds containment [0.02, 0.98]
  nudged.w = Math.min(0.96, Math.max(0.04, nudged.w));
  nudged.h = Math.min(0.96, Math.max(0.04, nudged.h));
  nudged.x = Math.max(0.02, Math.min(1.0 - nudged.w - 0.02, nudged.x));
  nudged.y = Math.max(0.02, Math.min(1.0 - nudged.h - 0.02, nudged.y));

  return nudged;
}

// ============================================================================
// Magnetic Snapping Engine (R4: +/-0.15s threshold)
// ============================================================================

/**
 * Snaps a target time to the nearest snap point if within threshold.
 * Default threshold is 0.15s per R4 specification.
 */
export function applyMagneticSnap(
  targetTime: number,
  snapPoints: number[],
  threshold = 0.15
): MagneticSnapResult {
  let closestDist = Infinity;
  let closestPoint: number | null = null;

  for (const point of snapPoints) {
    const dist = Math.abs(targetTime - point);
    if (dist < closestDist) {
      closestDist = dist;
      closestPoint = point;
    }
  }

  if (closestPoint !== null && closestDist <= threshold + 1e-6) {
    return {
      time: closestPoint,
      snapped: true,
      snapPoint: closestPoint,
      delta: closestDist,
    };
  }

  return { time: targetTime, snapped: false };
}
