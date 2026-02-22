import { X, Cpu, Zap, Clock, Minus, Sparkles } from 'lucide-react'

interface ApplianceEditorModalProps {
    showApplianceEditor: boolean;
    setShowApplianceEditor: (val: boolean) => void;
    applianceConfig: any[];
    setApplianceConfig: (val: any[]) => void;
    getCategory: (consumption: number) => { label: string; color: string; text: string };
    updateAppliance: (index: number, field: string, value: any) => void;
    removeAppliance: (index: number) => void;
    addAppliance: () => void;
    saveAppliances: () => void;
    isSaving: boolean;
}

export default function ApplianceEditorModal({
    showApplianceEditor,
    setShowApplianceEditor,
    applianceConfig,
    getCategory,
    updateAppliance,
    removeAppliance,
    addAppliance,
    saveAppliances,
    isSaving
}: ApplianceEditorModalProps) {
    if (!showApplianceEditor) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
            <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
                <div className="p-6 border-b flex items-center justify-between bg-gradient-to-r from-green-50 to-blue-50">
                    <div>
                        <h3 className="text-xl font-bold text-gray-900">Configure Appliances</h3>
                        <p className="text-sm text-gray-500">Add or edit appliances to personalize your energy optimization</p>
                    </div>
                    <button
                        onClick={() => setShowApplianceEditor(false)}
                        className="p-2 hover:bg-white/50 rounded-full transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-400 rotate-180" />
                    </button>
                </div>

                <div className="p-6 overflow-y-auto flex-1 space-y-6">
                    <div className="space-y-3">
                        <div className="flex items-center justify-between mb-2">
                            <h4 className="text-sm font-bold uppercase tracking-wider text-gray-400 flex items-center gap-2">
                                <Cpu className="w-4 h-4" />
                                Appliance Inventory
                            </h4>
                            <p className="text-[10px] text-gray-400 font-medium">Automatic classification based on kWh</p>
                        </div>

                        <div className="grid gap-3">
                            {applianceConfig?.map((app: any, idx: number) => {
                                const cat = getCategory(app.consumption_kwh);
                                return (
                                    <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 rounded-xl border border-gray-100 group transition-all hover:bg-white hover:shadow-sm">
                                        <div className="flex-1 grid grid-cols-12 gap-3 items-center">
                                            <div className="col-span-5">
                                                <input
                                                    className="w-full bg-transparent font-bold text-sm focus:ring-0 border-none p-0 text-gray-800 placeholder:text-gray-300"
                                                    value={app.name}
                                                    placeholder="Appliance Name"
                                                    onChange={(e) => updateAppliance(idx, 'name', e.target.value)}
                                                />
                                                <div className="flex items-center gap-1.5 mt-0.5">
                                                    <span className={`w-1.5 h-1.5 rounded-full ${cat.color}`} />
                                                    <span className={`text-[10px] font-bold uppercase tracking-tight ${cat.text}`}>{cat.label} Load</span>
                                                </div>
                                            </div>

                                            <div className="col-span-3 flex items-center gap-2 bg-white/50 px-2 py-1 rounded-lg border border-gray-100">
                                                <Zap className="w-3.5 h-3.5 text-yellow-500" />
                                                <input
                                                    type="number"
                                                    step="0.1"
                                                    className="w-full bg-transparent text-sm font-medium focus:ring-0 border-none p-0"
                                                    value={app.consumption_kwh}
                                                    onChange={(e) => updateAppliance(idx, 'consumption_kwh', parseFloat(e.target.value))}
                                                />
                                                <span className="text-[10px] text-gray-400 font-bold uppercase">kWh</span>
                                            </div>

                                            <div className="col-span-3 flex items-center gap-2 bg-white/50 px-2 py-1 rounded-lg border border-gray-100">
                                                <Clock className="w-3.5 h-3.5 text-blue-500" />
                                                <input
                                                    type="number"
                                                    className="w-full bg-transparent text-sm font-medium focus:ring-0 border-none p-0"
                                                    value={app.duration_hours}
                                                    onChange={(e) => updateAppliance(idx, 'duration_hours', parseInt(e.target.value))}
                                                />
                                                <span className="text-[10px] text-gray-400 font-bold uppercase">hrs</span>
                                            </div>
                                        </div>

                                        <button
                                            onClick={() => removeAppliance(idx)}
                                            className="p-2 hover:bg-red-50 text-red-300 hover:text-red-500 rounded-lg transition-all"
                                            title="Remove appliance"
                                        >
                                            <Minus className="w-4 h-4" />
                                        </button>
                                    </div>
                                );
                            })}

                            <button
                                onClick={addAppliance}
                                className="flex items-center justify-center gap-2 p-4 border-2 border-dashed border-gray-100 rounded-xl text-sm font-bold text-gray-400 hover:border-green-200 hover:text-green-600 hover:bg-green-50/30 transition-all active:scale-[0.98]"
                            >
                                <Sparkles className="w-4 h-4" />
                                Add New Appliance
                            </button>
                        </div>
                    </div>
                </div>

                <div className="p-6 border-t bg-gray-50 flex items-center justify-between">
                    <p className="text-xs text-gray-500">Changes will be saved to your profile and used for future tips.</p>
                    <div className="flex gap-3">
                        <button
                            onClick={() => setShowApplianceEditor(false)}
                            className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={saveAppliances}
                            disabled={isSaving}
                            className="px-6 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-bold rounded-xl shadow-lg shadow-green-200 transition-all active:scale-95 disabled:opacity-50"
                        >
                            {isSaving ? 'Saving...' : 'Save Configuration'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    )
}
