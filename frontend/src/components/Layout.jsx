import { Outlet, Link, useLocation } from "react-router-dom";
import { 
  LayoutDashboard, 
  Package, 
  Users, 
  FileText, 
  Wallet, 
  BarChart3, 
  Menu, 
  Settings,
  Download // ✅ أيقونة التحميل
} from "lucide-react";
import { useState } from "react";
import { systemAPI } from "../lib/api"; // ✅ استيراد API
import { toast } from "sonner"; // ✅ استيراد التنبيهات

export default function Layout() {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const navigation = [
    { name: 'لوحة التحكم', href: '/', icon: LayoutDashboard },
    { name: 'المنتجات', href: '/products', icon: Package },
    { name: 'العملاء', href: '/contacts', icon: Users },
    { name: 'الفواتير', href: '/invoices', icon: FileText },
    { name: 'الصندوق', href: '/cash', icon: Wallet },
    { name: 'التقارير', href: '/reports', icon: BarChart3 },
  ];

  const isActive = (path) => {
    if (path === '/' && location.pathname !== '/') return false;
    return location.pathname.startsWith(path);
  };

  // ✅ دالة معالجة تحميل ملف الإكسل
  const handleBackup = async () => {
    const toastId = toast.loading("جاري تصدير البيانات إلى Excel...");
    try {
      const response = await systemAPI.getBackup();
      
      // إنشاء رابط لتحميل الملف
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      
      const date = new Date().toISOString().slice(0, 10);
      link.setAttribute("download", `AlMoheet_Data_${date}.xlsx`);
      
      document.body.appendChild(link);
      link.click();
      
      link.remove();
      window.URL.revokeObjectURL(url);
      
      toast.success("تم تحميل ملف Excel بنجاح!", { id: toastId });
    } catch (error) {
      console.error(error);
      toast.error("فشل في تصدير البيانات", { id: toastId });
    }
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] flex font-sans" dir="rtl">
      
      {isMobileMenuOpen && (
        <div 
          className="fixed inset-0 bg-slate-900/20 z-40 lg:hidden backdrop-blur-sm transition-opacity"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      <aside className={`
        fixed lg:static inset-y-0 right-0 z-50 w-72 bg-white border-l border-slate-100/80 transition-transform duration-300 ease-in-out shadow-[0_0_40px_-10px_rgba(0,0,0,0.05)]
        ${isMobileMenuOpen ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
      `}>
        <div className="flex flex-col h-full">
          
          <div className="p-8 flex items-center gap-4">
            <div className="h-11 w-11 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-500 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-white">
              <span className="text-xl font-bold">م</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight text-slate-800">المُحيط</h1>
              <p className="text-xs text-slate-400 font-medium mt-0.5">برنامج المحاسبة</p>
            </div>
          </div>

          <nav className="flex-1 overflow-y-auto py-4 px-4 space-y-1.5">
            {navigation.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`
                    flex items-center gap-3.5 px-4 py-3 rounded-xl transition-all duration-200 group relative
                    ${active 
                      ? 'bg-emerald-50 text-emerald-700 font-bold shadow-sm ring-1 ring-emerald-100' 
                      : 'text-slate-500 hover:bg-slate-50 hover:text-slate-900 font-medium'
                    }
                  `}
                >
                  {active && <div className="absolute right-0 top-1/2 -translate-y-1/2 h-8 w-1 bg-emerald-500 rounded-l-full hidden"></div>}
                  <item.icon className={`h-5 w-5 transition-transform duration-200 ${active ? 'text-emerald-600 scale-110' : 'text-slate-400 group-hover:text-slate-600 group-hover:scale-105'}`} />
                  <span>{item.name}</span>
                  {active && <div className="mr-auto w-1.5 h-1.5 rounded-full bg-emerald-500"></div>}
                </Link>
              );
            })}
          </nav>

          {/* ✅ زر النسخ الاحتياطي المفعل */}
          <div className="p-4 mt-auto">
            <div 
              onClick={handleBackup}
              className="bg-slate-50 rounded-2xl p-3.5 flex items-center gap-3 border border-slate-100 hover:border-emerald-200 hover:bg-emerald-50/50 hover:shadow-sm transition-all cursor-pointer group"
              title="اضغط لتحميل نسخة احتياطية"
            >
              <div className="h-10 w-10 rounded-full bg-white flex items-center justify-center border border-slate-100 shadow-sm text-slate-400 group-hover:text-emerald-600 group-hover:border-emerald-100 transition-colors">
                <Settings className="h-5 w-5" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-slate-800 group-hover:text-emerald-900 transition-colors">نسخة احتياطية</p>
                <p className="text-xs text-slate-400 font-medium group-hover:text-emerald-600/80 transition-colors">حفظ البيانات (Excel)</p>
              </div>
              <button className="text-slate-400 group-hover:text-emerald-600 p-2 rounded-lg transition-colors">
                <Download className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        <header className="lg:hidden bg-white/80 backdrop-blur-md border-b border-slate-100 p-4 flex items-center justify-between sticky top-0 z-30">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 rounded-lg bg-emerald-500 flex items-center justify-center text-white font-bold shadow-sm">م</div>
            <span className="font-bold text-slate-900 text-lg">المُحيط</span>
          </div>
          <button onClick={() => setIsMobileMenuOpen(true)} className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors">
            <Menu className="h-6 w-6" />
          </button>
        </header>

        <main className="flex-1 overflow-auto p-4 md:p-8 scroll-smooth relative">
           <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none fixed"></div>
           <div className="max-w-7xl mx-auto relative z-10 space-y-8 pb-10">
            <Outlet />
           </div>
        </main>
      </div>
    </div>
  );
}