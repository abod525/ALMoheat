import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

// ✅ تنسيق العملة (أرقام إنجليزية + SP)
export function formatCurrency(amount) {
  // حماية من NaN: إذا لم يكن رقماً، نعتبره صفر
  const value = Number(amount) || 0;
  
  return new Intl.NumberFormat('en-US', {
    style: 'decimal', // نستخدم decimal لنضيف العملة يدوياً
    minimumFractionDigits: 0,
    maximumFractionDigits: 0, // الليرة عادة بدون قروش
  }).format(value) + " SP";
}

// ✅ تنسيق الأرقام العادية (أرقام إنجليزية)
export function formatNumber(number) {
  const value = Number(number) || 0;
  return new Intl.NumberFormat('en-US').format(value);
}

// ✅ تنسيق التاريخ (أرقام إنجليزية)
export function formatDate(date) {
  if (!date) return "";
  return new Date(date).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

// ✅ تمت إعادة الدالة المفقودة (لحل مشكلة التيرمينال)
export function formatDateTime(date) {
  if (!date) return "";
  return new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    hour12: true
  });
}