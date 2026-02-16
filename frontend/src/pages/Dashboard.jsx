import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { 
  TrendingUp, 
  Users, 
  Package, 
  Wallet, 
  ArrowUpRight, 
  ArrowDownRight,
  Calendar,
  Activity,
  FileText,
  DollarSign,
  AlertTriangle,
  CheckCircle2
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { reportsAPI } from "../lib/api";
import { formatCurrency } from "../lib/utils";
import { toast } from "sonner";

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await reportsAPI.getDashboard();
        setStats(response.data);
      } catch (error) {
        console.error("Dashboard error:", error);
        toast.error("فشل في تحميل البيانات");
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <div className="relative">
          <div className="h-16 w-16 rounded-full border-4 border-slate-200 border-t-slate-800 animate-spin"></div>
        </div>
      </div>
    );
  }

  const salesPercentage = stats.total_sales > 0 ? "+12.5%" : "0%";

  return (
    <div className="space-y-8 animate-fadeIn pb-10 font-sans">
      
      {/* 1. قسم الترحيب (Hero Section) */}
      <div className="relative overflow-hidden rounded-3xl bg-slate-900 p-8 text-white shadow-2xl ring-1 ring-white/10">
        <div className="absolute top-0 left-0 h-full w-full bg-gradient-to-br from-slate-800 to-slate-900 opacity-50"></div>
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-emerald-500 blur-[100px] opacity-20"></div>
        <div className="absolute -left-20 -bottom-20 h-64 w-64 rounded-full bg-blue-500 blur-[100px] opacity-20"></div>
        
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight mb-2">لوحة التحكم</h1>
            <p className="text-slate-400 max-w-lg">
              مرحباً بك في برنامج المُحيط..
            </p>
          </div>
          
          <div className="flex items-center gap-3 bg-white/10 backdrop-blur-md p-3 rounded-2xl border border-white/10">
            <div className="p-2 bg-emerald-500/20 rounded-lg">
              <Calendar className="h-6 w-6 text-emerald-400" />
            </div>
            <div>
              <p className="text-xs text-slate-300">تاريخ اليوم</p>
              <p className="font-semibold text-white font-mono" style={{ direction: 'ltr' }}>
                {new Date().toLocaleDateString('en-GB')}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. البطاقات الإحصائية العلوية */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        
        {/* رصيد الصندوق */}
        <Link to="/cash">
          <Card className="h-full border-none shadow-lg bg-slate-900 text-white hover:shadow-xl transition-all duration-300 hover:-translate-y-1 relative overflow-hidden group cursor-pointer rounded-3xl">
            <div className="absolute right-0 top-0 h-32 w-32 bg-white/10 rounded-full blur-3xl -mr-10 -mt-10 group-hover:bg-emerald-500/20 transition-colors"></div>
            <CardContent className="p-6 relative z-10">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-400 mb-1">رصيد الصندوق</p>
                  <h3 className="text-2xl font-bold text-white dir-ltr text-right">
                    {formatCurrency(stats.cash_balance)}
                  </h3>
                </div>
                {/* أيقونة متحركة (تكبير فقط لأن الخلفية غامقة) */}
                <div className="h-12 w-12 rounded-2xl bg-white/10 flex items-center justify-center backdrop-blur-sm border border-white/10 group-hover:scale-110 transition-transform duration-300 group-hover:bg-emerald-500 group-hover:border-emerald-400">
                  <Wallet className="h-6 w-6 text-emerald-400 group-hover:text-white transition-colors" />
                </div>
              </div>
              <div className="mt-4 flex items-center gap-2 text-xs text-slate-300">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                محدث الآن
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* المبيعات */}
        <Link to="/invoices">
          <Card className="h-full border-none shadow-sm bg-white hover:shadow-xl transition-all duration-300 hover:-translate-y-1 group cursor-pointer rounded-3xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-500 mb-1">إجمالي المبيعات</p>
                  <h3 className="text-2xl font-bold text-slate-900">{formatCurrency(stats.total_sales)}</h3>
                </div>
                {/* تأثير التلوين عند التمرير */}
                <div className="h-12 w-12 rounded-2xl bg-emerald-50 flex items-center justify-center text-emerald-600 group-hover:bg-emerald-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-emerald-200 group-hover:scale-110">
                  <TrendingUp className="h-6 w-6" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-xs">
                <span className="text-emerald-700 bg-emerald-50 px-2 py-1 rounded-md font-bold flex items-center gap-1 group-hover:bg-emerald-100 transition-colors">
                  <ArrowUpRight className="h-3 w-3" /> {salesPercentage}
                </span>
                <span className="text-slate-400 mr-2">مقارنة بالشهر الماضي</span>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* العملاء */}
        <Link to="/contacts">
          <Card className="h-full border-none shadow-sm bg-white hover:shadow-xl transition-all duration-300 hover:-translate-y-1 group cursor-pointer rounded-3xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-500 mb-1">العملاء</p>
                  <h3 className="text-2xl font-bold text-slate-900">{stats.customers_count}</h3>
                </div>
                {/* تأثير التلوين عند التمرير */}
                <div className="h-12 w-12 rounded-2xl bg-blue-50 flex items-center justify-center text-blue-600 group-hover:bg-blue-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-blue-200 group-hover:scale-110">
                  <Users className="h-6 w-6" />
                </div>
              </div>
              <div className="mt-4 flex items-center text-xs">
                <span className="text-blue-700 bg-blue-50 px-2 py-1 rounded-md font-bold group-hover:bg-blue-100 transition-colors">
                  عملاء نشطون
                </span>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* المنتجات */}
        <Link to="/products">
          <Card className="h-full border-none shadow-sm bg-white hover:shadow-xl transition-all duration-300 hover:-translate-y-1 group cursor-pointer rounded-3xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-500 mb-1">المنتجات</p>
                  <h3 className="text-2xl font-bold text-slate-900">{stats.products_count}</h3>
                </div>
                {/* تأثير التلوين عند التمرير */}
                <div className="h-12 w-12 rounded-2xl bg-indigo-50 flex items-center justify-center text-indigo-600 group-hover:bg-indigo-500 group-hover:text-white transition-all duration-300 shadow-sm group-hover:shadow-indigo-200 group-hover:scale-110">
                  <Package className="h-6 w-6" />
                </div>
              </div>
              <div className="mt-4 text-xs text-slate-500">
                منتجات مسجلة في النظام
              </div>
            </CardContent>
          </Card>
        </Link>
      </div>

      {/* 3. القسم السفلي: الجدول (يمين) + الملخصات (يسار) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
        
        {/* جدول أحدث الفواتير (يأخذ ثلثين الشاشة) */}
        <Card className="md:col-span-2 border-none shadow-sm bg-white overflow-hidden rounded-3xl">
          <CardHeader className="bg-white border-b border-slate-100 pb-4 pt-6 px-6">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg font-bold text-slate-800 flex items-center gap-2">
                <div className="p-2 bg-indigo-50 rounded-lg">
                  <DollarSign className="h-5 w-5 text-indigo-600" />
                </div>
                أحدث الفواتير
              </CardTitle>
              <Link to="/invoices">
                <Button variant="ghost" size="sm" className="text-slate-500 hover:text-indigo-600 hover:bg-indigo-50">
                  عرض الكل
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent className="p-0">
            {stats.recent_invoices && stats.recent_invoices.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-right">
                  <thead className="bg-slate-50/50 text-slate-500 font-medium">
                    <tr>
                      <th className="px-6 py-4">رقم الفاتورة</th>
                      <th className="px-6 py-4">العميل</th>
                      <th className="px-6 py-4">المبلغ</th>
                      <th className="px-6 py-4">الحالة</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {stats.recent_invoices.map((invoice) => (
                      <tr key={invoice.id} className="hover:bg-slate-50/80 transition-colors group cursor-pointer">
                        <td className="px-6 py-4 font-medium text-slate-700 group-hover:text-indigo-600 transition-colors">
                          {invoice.invoice_number}
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-slate-600 group-hover:text-slate-900 transition-colors">{invoice.contact_name}</span>
                        </td>
                        <td className="px-6 py-4 font-bold text-slate-800">{formatCurrency(invoice.total)}</td>
                        <td className="px-6 py-4">
                          <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${
                            invoice.status === 'paid' ? 'bg-emerald-50 text-emerald-700 border-emerald-100 group-hover:bg-emerald-100' :
                            invoice.status === 'cancelled' ? 'bg-red-50 text-red-700 border-red-100 group-hover:bg-red-100' :
                            'bg-amber-50 text-amber-700 border-amber-100 group-hover:bg-amber-100'
                          } transition-colors`}>
                            {invoice.status === 'paid' ? 'مدفوعة' : invoice.status === 'cancelled' ? 'ملغاة' : 'معلقة'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-12 text-center flex flex-col items-center justify-center text-slate-400">
                <FileText className="h-12 w-12 mb-3 opacity-20" />
                <p>لا توجد فواتير حديثة</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* العمود الجانبي: الأداء المالي + التنبيهات */}
        <div className="flex flex-col gap-6">
          
          {/* بطاقة الأداء المالي */}
          <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden hover:shadow-md transition-shadow">
            <CardHeader className="bg-white border-b border-slate-100 pb-4 pt-6 px-6">
              <CardTitle className="text-lg font-bold text-slate-800">الأداء المالي</CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-4">
              <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-100 transition-all hover:bg-emerald-100/50 cursor-default group">
                <div className="flex items-center gap-3 mb-1">
                  <div className="p-1.5 bg-emerald-200/50 rounded-md group-hover:bg-emerald-500 group-hover:text-white transition-colors">
                    <ArrowUpRight className="h-4 w-4 text-emerald-700 group-hover:text-white" />
                  </div>
                  <span className="text-sm font-medium text-emerald-800">إجمالي المبيعات</span>
                </div>
                <p className="text-2xl font-bold text-emerald-700">{formatCurrency(stats.total_sales)}</p>
              </div>

              <div className="p-4 rounded-2xl bg-blue-50 border border-blue-100 transition-all hover:bg-blue-100/50 cursor-default group">
                <div className="flex items-center gap-3 mb-1">
                  <div className="p-1.5 bg-blue-200/50 rounded-md group-hover:bg-blue-500 group-hover:text-white transition-colors">
                    <ArrowDownRight className="h-4 w-4 text-blue-700 group-hover:text-white" />
                  </div>
                  <span className="text-sm font-medium text-blue-800">إجمالي المشتريات</span>
                </div>
                <p className="text-2xl font-bold text-blue-700">{formatCurrency(stats.total_purchases)}</p>
              </div>

              <div className="pt-4 mt-2 border-t border-slate-100">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-500 font-medium">صافي الربح التقديري</span>
                  <span className={`text-lg font-bold ${stats.total_sales - stats.total_purchases >= 0 ? 'text-slate-800' : 'text-red-600'}`}>
                    {formatCurrency(stats.total_sales - stats.total_purchases)}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* بطاقة التنبيهات */}
          <Link to="/products">
            <Card className="border-none shadow-sm bg-white rounded-3xl overflow-hidden hover:shadow-md transition-shadow cursor-pointer group">
              <CardHeader className="bg-white border-b border-slate-100 pb-4 pt-6 px-6">
                <CardTitle className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-amber-500 group-hover:scale-110 transition-transform" />
                  التنبيهات
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                {stats.low_stock_count > 0 ? (
                  <div className="p-4 rounded-2xl bg-amber-50 border border-amber-100 flex items-start gap-3 group-hover:bg-amber-100/50 transition-colors">
                    <div className="p-1.5 bg-amber-200/50 rounded-full mt-0.5 group-hover:bg-amber-500 group-hover:text-white transition-colors">
                      <AlertTriangle className="h-4 w-4 text-amber-700 group-hover:text-white" />
                    </div>
                    <div>
                      <p className="text-amber-800 font-bold text-lg">{stats.low_stock_count} منتج</p>
                      <p className="text-amber-700 text-sm">وصلت للحد الأدنى من المخزون.</p>
                    </div>
                  </div>
                ) : (
                  <div className="p-4 rounded-2xl bg-emerald-50 border border-emerald-100 flex items-center gap-3 group-hover:bg-emerald-100/50 transition-colors">
                    <div className="p-1.5 bg-emerald-200/50 rounded-full group-hover:bg-emerald-500 group-hover:text-white transition-colors">
                      <CheckCircle2 className="h-4 w-4 text-emerald-700 group-hover:text-white" />
                    </div>
                    <p className="text-emerald-800 font-medium">المخزون ممتاز، لا توجد نواقص.</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </Link>

        </div>
      </div>
    </div>
  );
}